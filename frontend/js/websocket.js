/**
 * WebSocket module — connection management and message routing.
 */

class VoiceWebSocket {
    constructor() {
        this.ws = null;
        this.onStateChange = null;   // (state: string) => void
        this.onTranscript = null;    // (role, text, final) => void
        this.onInterrupt = null;     // () => void
        this.onTTSAudio = null;      // (ArrayBuffer) => void
        this.onConnectionChange = null; // (connected: boolean) => void
        this.onTTSConfig = null;     // (sampleRate: number) => void
        this.onError = null;         // (message: string) => void
    }

    connect() {
        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        const url = `${protocol}//${location.host}/ws`;

        this.ws = new WebSocket(url);
        this.ws.binaryType = "arraybuffer";

        this.ws.onopen = () => {
            console.log("WebSocket connected");
            // Send config
            this.ws.send(JSON.stringify({
                type: "config",
                sampleRate: 16000,
            }));
            if (this.onConnectionChange) this.onConnectionChange(true);
        };

        this.ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                // Binary = TTS audio
                if (this.onTTSAudio) this.onTTSAudio(event.data);
                return;
            }

            // JSON control message
            const msg = JSON.parse(event.data);

            switch (msg.type) {
                case "state":
                    if (this.onStateChange) this.onStateChange(msg.state);
                    break;
                case "transcript":
                    if (this.onTranscript)
                        this.onTranscript(msg.role, msg.text, msg.final);
                    break;
                case "interrupt":
                    if (this.onInterrupt) this.onInterrupt();
                    break;
                case "tts_config":
                    if (this.onTTSConfig) this.onTTSConfig(msg.sampleRate);
                    break;
                case "error":
                    if (this.onError) this.onError(msg.message);
                    break;
            }
        };

        this.ws.onclose = () => {
            console.log("WebSocket disconnected");
            if (this.onConnectionChange) this.onConnectionChange(false);
        };

        this.ws.onerror = (err) => {
            console.error("WebSocket error:", err);
        };
    }

    sendAudio(int16Array) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(int16Array.buffer);
        }
    }

    sendInterrupt() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "interrupt" }));
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}
