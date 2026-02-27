/**
 * Audio module — mic capture (PCM 16-bit 16kHz) and TTS playback.
 */

class AudioManager {
    constructor() {
        this.audioContext = null;
        this.stream = null;
        this.workletNode = null;
        this.onAudioChunk = null; // callback: (Int16Array) => void
        this.onAudioLevel = null; // callback: (0..1) => void

        // TTS playback queue
        this._playbackQueue = [];
        this._isPlaying = false;
        this._playbackSampleRate = 22050; // Piper default
    }

    async startMic() {
        this.audioContext = new AudioContext({ sampleRate: 16000 });

        this.stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        // Load AudioWorklet for PCM extraction
        await this.audioContext.audioWorklet.addModule("/js/pcm-worklet.js");
        const source = this.audioContext.createMediaStreamSource(this.stream);
        this.workletNode = new AudioWorkletNode(this.audioContext, "pcm-processor");

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
        // Don't connect to destination — we don't want mic echoed to speakers
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
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
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

    async _playNext() {
        if (this._playbackQueue.length === 0) {
            this._isPlaying = false;
            return;
        }

        this._isPlaying = true;
        const pcmBuffer = this._playbackQueue.shift();

        // Convert Int16 PCM to Float32 for Web Audio API
        const int16 = new Int16Array(pcmBuffer);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768.0;
        }

        // Create AudioBuffer and play
        const playbackCtx = this.audioContext || new AudioContext();
        const audioBuffer = playbackCtx.createBuffer(
            1,
            float32.length,
            this._playbackSampleRate
        );
        audioBuffer.getChannelData(0).set(float32);

        const source = playbackCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(playbackCtx.destination);
        source.onended = () => this._playNext();
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
                this._currentSource.stop();
            } catch (e) {
                // Already stopped
            }
            this._currentSource = null;
        }
    }
}
