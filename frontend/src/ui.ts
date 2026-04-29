const chatLog = document.getElementById("chat-log") as HTMLDivElement;
const input = document.getElementById("user-input") as HTMLTextAreaElement;
const submitBtn = document.getElementById("submit-btn") as HTMLButtonElement;
const spinner = document.getElementById("spinner") as HTMLDivElement;

export function appendUserMessage(text: string): void {
  const msg = document.createElement("div");
  msg.className = "message user";
  msg.textContent = text;
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
}

export function appendBotMessage(text: string): void {
  const msg = document.createElement("div");
  msg.className = "message bot";
  msg.textContent = text;
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
}

export function appendErrorMessage(text: string): void {
  const msg = document.createElement("div");
  msg.className = "message error";
  msg.textContent = text;
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
}

export function setLoading(loading: boolean): void {
  spinner.style.display = loading ? "block" : "none";
  submitBtn.disabled = loading;
  input.disabled = loading;
}

export function clearInput(): void {
  input.value = "";
}

export function getInputValue(): string {
  return input.value.trim();
}
