# ai-voice-assistant

The AI voice assistant demo of the paper, "Building an intelligent voice assistant using machine learning: A study of the technology and practical applications of speech processing".

## Usage

Install [rye](https://github.com/astral-sh/rye) first.

Then, install dependencies and create virtual environment.

```bash
rye sync
```

Finally, configure your OpenAI API key in the environment variable,
`OPENAI_API_KEY`, and run the demo.

```bash
export OPENAI_API_KEY=your-api-key
rye run ai-voice-assistant
```

## License

AGPL-3.0-or-later
