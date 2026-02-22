/**
 * LLM proxy for FER_LLM app.
 * Architecture: Python app → this server → Duke API.
 * The app only talks to this proxy; it never sees the Duke URL or API key.
 * Only this server needs to reach Duke (Duke VPN on this machine if required).
 */

const express = require('express');

// Duke URL and key live only on the server; client never sends them.
const LLM_API_URL = process.env.DUKE_LLM_URL || 'https://litellm.oit.duke.edu/v1/chat/completions';
const LLM_API_KEY = process.env.DUKE_LLM_KEY || 'sk-dwAYbKw4KalzudSkQVcOWg';
const PORT = Number(process.env.PORT) || 3001;

const app = express();
app.use(express.json());

app.post('/proxy/llm', async (req, res) => {
  try {
    const { model, messages } = req.body;
    console.log('LLM proxy: request for model:', model || '(default)');

    const response = await fetch(LLM_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${LLM_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: model || 'gpt-4o',
        messages: messages || [],
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('LLM proxy: API error', response.status, errorText);
      res.status(response.status).json({
        error: 'Duke LLM API error',
        status: response.status,
        details: errorText,
      });
      return;
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('LLM proxy error:', error.message);
    res.status(500).json({
      error: 'Failed to reach Duke LLM API',
      details: error.message,
    });
  }
});

app.listen(PORT, () => {
  console.log(`LLM proxy listening on http://localhost:${PORT}`);
  console.log(`  POST /proxy/llm -> ${LLM_API_URL}`);
});
