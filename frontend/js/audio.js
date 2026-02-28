/**
 * Audio module — mic capture (PCM 16-bit 16kHz) and TTS playback.
 *
 * Uses separate AudioContexts for mic (16kHz) and playback (22050Hz)
 * to avoid sample rate conflicts.
 *
 * Client-side interrupt detection: monitors raw mic level during playback.
 * If the user speaks loudly while TTS is playing, triggers onUserSpeaking
 * callback so the app can send an interrupt signal to the server.
 * This is more reliable than server-side detection because browser echo
 * cancellation can filter out the user's voice from the audio stream.
 */

// Interrupt detection config
const INTERRUPT_LEVEL_THRESHOLD = 0.03; // raw RMS level to consider "user speaking"
const INTERRUPT_CONSECUTIVE = 4;         // consecutive loud chunks needed

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

        // Client-side interrupt detection
        this._interruptCount = 0;
        this._interruptCooldown = false;
    }

    get isPlayingTTS() {
        return this._isPlaying;
    }

    setPlaybackSampleRate(rate) {
        if (rate === this._playbackSampleRate) return;
        this._playbackSampleRate = rate;
        // Recreate playback context if already running
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

            if (this.onAudioChunk) {
                this.onAudioChunk(new Int16Array(pcm));
            }
            if (this.onAudioLevel) {
                this.onAudioLevel(level);
            }

            // Client-side interrupt detection: check raw mic level during playback
            if (this._isPlaying && !this._interruptCooldown) {
                if (level > INTERRUPT_LEVEL_THRESHOLD) {
                    this._interruptCount++;
                    if (this._interruptCount >= INTERRUPT_CONSECUTIVE) {
                        this._interruptCount = 0;
                        this._triggerInterrupt();
                    }
                } else {
                    this._interruptCount = 0;
                }
            }
        };

        source.connect(this.workletNode);

        // Playback context at TTS sample rate
        this.playbackContext = new AudioContext({ sampleRate: this._playbackSampleRate });
        this._gainNode = this.playbackContext.createGain();
        this._gainNode.connect(this.playbackContext.destination);
    }

    _triggerInterrupt() {
        console.log("Client-side interrupt: user speaking during playback");
        this.flushPlayback();
        // Cooldown: don't re-trigger for 1 second
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
            this._playNext();
        }
    }

    _playNext() {
        if (this._playbackQueue.length === 0 || !this.playbackContext) {
            this._isPlaying = false;
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
