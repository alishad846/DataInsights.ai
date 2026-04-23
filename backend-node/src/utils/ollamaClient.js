import fetch from "node-fetch";

export const askOllama = async (prompt) => {
  try {
    const response = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "phi3:mini",
        prompt,
        stream: false
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      if (errorText.toLowerCase().includes("does not support image") || 
          errorText.toLowerCase().includes("cannot read image")) {
        throw new Error("Model does not support image input. Please use a text-only model.");
      }
      throw new Error(`Ollama request failed: ${response.status}`);
    }

    const data = await response.json();
    const responseText = data.response || "";
    
    if (responseText.toLowerCase().includes("does not support image") || 
        responseText.toLowerCase().includes("cannot read image")) {
      throw new Error("Model does not support image input. Please use a text-only model.");
    }
    
    return responseText;
  } catch (err) {
    if (err.message.includes("does not support image") || 
        err.message.includes("Cannot read image")) {
      throw err;
    }
    throw new Error(`Ollama connection error: ${err.message}`);
  }
};
