/**
 * Main app — wires AudioManager, VoiceWebSocket, and DOM together.
 */

const audio = new AudioManager();
const socket = new VoiceWebSocket();

// DOM elements
const micBtn = document.getElementById("mic-btn");
const statusEl = document.getElementById("status");
const stateEl = document.getElementById("state-indicator");
const convoEl = document.getElementById("conversation");
const levelBar = document.getElementById("audio-level-bar");

let isActive = false;
let currentAssistantEl = null;

// --- Socket callbacks ---

socket.onConnectionChange = (connected) => {
    statusEl.textContent = connected ? "Online" : "Offline";
    statusEl.className = `status ${connected ? "online" : "offline"}`;
    micBtn.disabled = !connected;
};

socket.onStateChange = (state) => {
    stateEl.textContent = state;
    stateEl.className = `state-indicator ${state.toLowerCase()}`;
};

socket.onTranscript = (role, text, final) => {
    if (role === "user") {
        addMessage("user", text);
    } else if (role === "assistant") {
        if (!currentAssistantEl) {
            currentAssistantEl = addMessage("assistant", text, !final);
        } else {
            currentAssistantEl.textContent = text;
            if (final) {
                currentAssistantEl.classList.remove("streaming");
                currentAssistantEl = null;
            }
        }
    }
};

socket.onTTSAudio = (buffer) => {
    audio.queueTTSAudio(buffer);
};

socket.onInterrupt = () => {
    audio.flushPlayback();
    if (currentAssistantEl) {
        currentAssistantEl.classList.remove("streaming");
        currentAssistantEl = null;
    }
};

socket.onError = (msg) => {
    console.error("Server error:", msg);
};

// --- Audio callbacks ---

audio.onAudioChunk = (int16Array) => {
    socket.sendAudio(int16Array);
};

audio.onAudioLevel = (level) => {
    const pct = Math.min(level * 500, 100);
    levelBar.style.width = `${pct}%`;
};

// Client-side interrupt: user spoke during TTS playback
audio.onUserSpeaking = () => {
    socket.sendInterrupt();
    if (currentAssistantEl) {
        currentAssistantEl.classList.remove("streaming");
        currentAssistantEl = null;
    }
};

// --- UI ---

function addMessage(role, text, streaming = false) {
    const el = document.createElement("div");
    el.className = `message ${role}${streaming ? " streaming" : ""}`;
    el.textContent = text;
    convoEl.appendChild(el);
    convoEl.scrollTop = convoEl.scrollHeight;
    return el;
}

micBtn.addEventListener("click", async () => {
    if (!isActive) {
        try {
            await audio.startMic();
            socket.connect();
            micBtn.textContent = "Stop";
            micBtn.classList.add("active");
            isActive = true;
        } catch (err) {
            console.error("Failed to start:", err);
            alert("Microphone access required.");
        }
    } else {
        audio.stopMic();
        socket.disconnect();
        micBtn.textContent = "Start";
        micBtn.classList.remove("active");
        isActive = false;
        levelBar.style.width = "0%";
        stateEl.textContent = "IDLE";
        stateEl.className = "state-indicator";
    }
});
