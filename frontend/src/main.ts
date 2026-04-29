import { postRecommend } from "./api.js";
import {
  appendBotMessage,
  appendErrorMessage,
  appendUserMessage,
  clearInput,
  getInputValue,
  setLoading,
} from "./ui.js";

const submitBtn = document.getElementById("submit-btn") as HTMLButtonElement;
const inputEl = document.getElementById("user-input") as HTMLTextAreaElement;

async function handleSubmit(): Promise<void> {
  const text = getInputValue();
  if (!text) return;

  appendUserMessage(text);
  clearInput();
  setLoading(true);

  try {
    const result = await postRecommend(text);

    if (result.error === "DRINK_NOT_FOUND") {
      appendErrorMessage(
        "Sorry, we couldn't find that drink in our system. Try a Dutch Bros drink like Golden Eagle, Annihilator, or Caramelizer."
      );
    } else if (result.error) {
      appendErrorMessage("Something went wrong. Please try again.");
    } else {
      appendBotMessage(result.response);
    }
  } catch {
    appendErrorMessage("Could not reach the server. Make sure the backend is running.");
  } finally {
    setLoading(false);
  }
}

submitBtn.addEventListener("click", handleSubmit);

inputEl.addEventListener("keydown", (e: KeyboardEvent) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSubmit();
  }
});
