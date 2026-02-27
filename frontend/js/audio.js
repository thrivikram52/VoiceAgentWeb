/**
 * Audio module — mic capture (PCM 16-bit 16kHz) and TTS playback.
 *
 * Uses separate AudioContexts for mic (16kHz) and playback (22050Hz)
 * to avoid sample rate conflicts.
 */

class AudioManager {
    constructor() {
        // Mic capture
        this.micContext = null;
        this.stream = null;
        this.workletNode = null;
        this.onAudioChunk = null; // callback: (Int16Array) => void
        this.onAudioLevel = null; // callback: (0..1) => void

        // TTS playback — separate context at TTS sample rate
        this.playbackContext = null;
        this._playbackQueue = [];
        this._isPlaying = false;
        this._currentSource = null;
        this._gainNode = null;
        this._playbackSampleRate = 22050; // Piper default
    }

    async startMic() {
        // Mic context at 16kHz for whisper.cpp
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
        };

        source.connect(this.workletNode);

        // Playback context at TTS sample rate
        this.playbackContext = new AudioContext({ sampleRate: this._playbackSampleRate });
        this._gainNode = this.playbackContext.createGain();
        this._gainNode.connect(this.playbackContext.destination);
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

    /**
     * Queue TTS audio (Int16 PCM) for playback.
     * @param {ArrayBuffer} pcmBuffer - Raw PCM bytes (int16)
     */
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

    /**
     * Stop all TTS playback immediately (for barge-in).
     */
    flushPlayback() {
        this._playbackQueue = [];
        this._isPlaying = false;
        if (this._currentSource) {
            try {
                this._currentSource.onended = null; // prevent chain
                this._currentSource.stop();
                this._currentSource.disconnect();
            } catch (e) {
                // Already stopped
            }
            this._currentSource = null;
        }
    }
}
