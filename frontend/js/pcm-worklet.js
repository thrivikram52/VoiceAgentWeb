/**
 * AudioWorklet processor — converts float32 audio to int16 PCM chunks.
 * Runs in a separate thread for low-latency processing.
 */

class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = new Float32Array(0);
        // Send chunks of ~32ms at 16kHz = 512 samples
        this._chunkSize = 512;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channelData = input[0];

        // Append to buffer
        const newBuffer = new Float32Array(
            this._buffer.length + channelData.length
        );
        newBuffer.set(this._buffer);
        newBuffer.set(channelData, this._buffer.length);
        this._buffer = newBuffer;

        // Send complete chunks
        while (this._buffer.length >= this._chunkSize) {
            const chunk = this._buffer.slice(0, this._chunkSize);
            this._buffer = this._buffer.slice(this._chunkSize);

            // Convert to Int16
            const pcm = new Int16Array(chunk.length);
            for (let i = 0; i < chunk.length; i++) {
                const s = Math.max(-1, Math.min(1, chunk[i]));
                pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            }

            // Calculate RMS level for visualization
            let sum = 0;
            for (let i = 0; i < chunk.length; i++) {
                sum += chunk[i] * chunk[i];
            }
            const level = Math.sqrt(sum / chunk.length);

            this.port.postMessage(
                { pcm: pcm.buffer, level },
                [pcm.buffer]
            );
        }

        return true;
    }
}

registerProcessor("pcm-processor", PCMProcessor);
