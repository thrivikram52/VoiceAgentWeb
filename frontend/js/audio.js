/**
 * Audio module — mic capture (PCM 16-bit 16kHz) and TTS playback.
 *
 * Uses separate AudioContexts for mic (16kHz) and playback (22050Hz)
 * to avoid sample rate conflicts.
 *
 * Client-side interrupt detection: uses adaptive echo baseline.
 * Tracks the average mic level during TTS playback (echo floor),
 * then triggers interrupt only when mic level spikes well above
 * that baseline — indicating the user is actually speaking.
 */

// Interrupt detection config
const INTERRUPT_SPIKE_FACTOR = 3.0;   // mic level must be 3x above echo baseline
const INTERRUPT_MIN_LEVEL = 0.05;     // absolute minimum level to even consider
const INTERRUPT_CONSECUTIVE = 10;     // consecutive spike chunks needed
const INTERRUPT_BASELINE_WARMUP = 15; // chunks to collect before enabling detection

class AudioManager {
    constructor() {
        // Mic capture
        this.micContext = null;
        this.stream = null;
        this.workletNode = null;
        this.onAudioChunk = null;    // (Int16Array) => void
        this.onAudioLevel = null;    // (0..1) => void
        this.onUserSpeaking = null;  // () => void — fired when user speaks during playback

        // TTS playback — separate context at TTS sample rate
        this.playbackContext = null;
        this._playbackQueue = [];
        this._isPlaying = false;
        this._currentSource = null;
        this._gainNode = null;
        this._playbackSampleRate = 22050;

        // Client-side interrupt detection (adaptive)
        this._interruptCount = 0;
        this._interruptCooldown = false;
        this._echoBaseline = 0;       // running average of mic level during playback
        this._echoSamples = 0;        // number of samples in the baseline

        // Post-playback grace period to let echo die out
        this._playbackEndTime = 0;
    }

    get isPlayingOrEchoing() {
        return this._isPlaying || (Date.now() - this._playbackEndTime < 1500);
    }

    get isPlayingTTS() {
        return this._isPlaying;
    }

    setPlaybackSampleRate(rate) {
        if (rate === this._playbackSampleRate) return;
        this._playbackSampleRate = rate;
        if (this.playbackContext) {
            this.flushPlayback();
            this.playbackContext.close();
            this.playbackContext = new AudioContext({ sampleRate: rate });
            this._gainNode = this.playbackContext.createGain();
            this._gainNode.connect(this.playbackContext.destination);
        }
    }

    async startMic() {
        this.micContext = new AudioContext({ sampleRate: 16000 });

        this.stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        await this.micContext.audioWorklet.addModule("/js/pcm-worklet.js");
        const source = this.micContext.createMediaStreamSource(this.stream);
        this.workletNode = new AudioWorkletNode(this.micContext, "pcm-processor");

        this.workletNode.port.onmessage = (event) => {
            const { pcm, level } = event.data;

            if (this.isPlayingOrEchoing) {
                // During TTS playback (+ echo grace period): don't send audio to server.
                if (this.onAudioLevel) {
                    this.onAudioLevel(level);
                }

                // Adaptive interrupt detection during actual playback only
                if (this._isPlaying && !this._interruptCooldown) {
                    this._updateEchoBaseline(level);

                    // Only check after warmup period (baseline needs time to stabilize)
                    if (this._echoSamples >= INTERRUPT_BASELINE_WARMUP) {
                        const threshold = Math.max(
                            this._echoBaseline * INTERRUPT_SPIKE_FACTOR,
                            INTERRUPT_MIN_LEVEL
                        );
                        if (level > threshold) {
                            this._interruptCount++;
                            if (this._interruptCount >= INTERRUPT_CONSECUTIVE) {
                                this._interruptCount = 0;
                                this._triggerInterrupt();
                            }
                        } else {
                            this._interruptCount = 0;
                        }
                    }
                }
                return;
            }

            // Not playing: send audio to server for VAD/STT
            if (this.onAudioChunk) {
                this.onAudioChunk(new Int16Array(pcm));
            }
            if (this.onAudioLevel) {
                this.onAudioLevel(level);
            }
        };

        source.connect(this.workletNode);

        // Playback context at TTS sample rate
        this.playbackContext = new AudioContext({ sampleRate: this._playbackSampleRate });
        this._gainNode = this.playbackContext.createGain();
        this._gainNode.connect(this.playbackContext.destination);
    }

    _updateEchoBaseline(level) {
        // Exponential moving average — adapts to current echo level
        // Only incorporate levels that aren't obvious spikes (< 4x current baseline)
        // This prevents real speech from polluting the echo baseline
        if (this._echoSamples === 0) {
            this._echoBaseline = level;
            this._echoSamples = 1;
        } else if (level < this._echoBaseline * 4 || this._echoSamples < INTERRUPT_BASELINE_WARMUP) {
            const alpha = 0.1; // smoothing factor
            this._echoBaseline = this._echoBaseline * (1 - alpha) + level * alpha;
            this._echoSamples++;
        }
        // If level is a huge spike (>4x baseline), don't update — it's likely speech
    }

    _resetEchoBaseline() {
        this._echoBaseline = 0;
        this._echoSamples = 0;
        this._interruptCount = 0;
    }

    _triggerInterrupt() {
        console.log("Client-side interrupt: user speaking during playback");
        this.flushPlayback();
        this._interruptCooldown = true;
        setTimeout(() => { this._interruptCooldown = false; }, 1000);
        if (this.onUserSpeaking) {
            this.onUserSpeaking();
        }
    }

    stopMic() {
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach((t) => t.stop());
            this.stream = null;
        }
        if (this.micContext) {
            this.micContext.close();
            this.micContext = null;
        }
        this.flushPlayback();
        if (this.playbackContext) {
            this.playbackContext.close();
            this.playbackContext = null;
        }
    }

    queueTTSAudio(pcmBuffer) {
        this._playbackQueue.push(pcmBuffer);
        if (!this._isPlaying) {
            this._resetEchoBaseline();
            this._playNext();
        }
    }

    _playNext() {
        if (this._playbackQueue.length === 0 || !this.playbackContext) {
            this._isPlaying = false;
            this._playbackEndTime = Date.now();
            return;
        }

        this._isPlaying = true;
        const pcmBuffer = this._playbackQueue.shift();

        const int16 = new Int16Array(pcmBuffer);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768.0;
        }

        const audioBuffer = this.playbackContext.createBuffer(
            1,
            float32.length,
            this._playbackSampleRate
        );
        audioBuffer.getChannelData(0).set(float32);

        const source = this.playbackContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this._gainNode);
        source.onended = () => {
            if (this._currentSource === source) {
                this._currentSource = null;
                this._playNext();
            }
        };
        source.start();
        this._currentSource = source;
    }

    flushPlayback() {
        this._playbackQueue = [];
        this._isPlaying = false;
        this._playbackEndTime = Date.now();
        this._interruptCount = 0;
        if (this._currentSource) {
            try {
                this._currentSource.onended = null;
                this._currentSource.stop();
                this._currentSource.disconnect();
            } catch (e) {
                // Already stopped
            }
            this._currentSource = null;
        }
    }
}
