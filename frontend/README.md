# Zuora Help Agent - Frontend

Minimal, clean chat interface for the Zuora Help Agent.

## Features

- ✅ Simple HTML + CSS + JavaScript (no frameworks)
- ✅ Clean, modern chat interface
- ✅ Connects to FastAPI `/ask` endpoint
- ✅ Shows answers with clickable sources
- ✅ Typing indicators and loading states
- ✅ Error handling
- ✅ Responsive design
- ✅ Conversation persistence (session-based)

## Quick Start

### Option 1: Using Python HTTP Server

```bash
cd frontend
python -m http.server 3000
```

Then open: http://localhost:3000

### Option 2: Using Node.js HTTP Server

```bash
cd frontend
npx http-server -p 3000
```

Then open: http://localhost:3000

### Option 3: Open Directly

Simply open `index.html` in your browser (file:// protocol).

**Note:** If using file://, you may need to enable CORS in the backend or use a local server.

## Backend Setup

The frontend expects the backend to be running at `http://localhost:8000`.

Start the backend:

```bash
cd backend
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

Edit `app.js` to change the API endpoint:

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

## File Structure

```
frontend/
├── index.html      # Main HTML structure
├── style.css       # Styles and layout
├── app.js          # JavaScript logic
├── serve.py        # Simple Python server
└── README.md       # This file
```

## API Integration

The frontend calls the `/ask` endpoint:

**Request:**
```json
{
  "question": "What is Zuora CPQ?",
  "conversation_id": "conv_1234567890_abc123",
  "context": {}
}
```

**Response:**
```json
{
  "answer": "Zuora CPQ is a configure, price, quote solution...",
  "sources": [
    {
      "title": "Zuora CPQ Overview",
      "source": "zuora_cpq_guide.md",
      "url": "https://docs.zuora.com/cpq"
    }
  ],
  "conversation_id": "conv_1234567890_abc123",
  "confidence": 0.92
}
```

## Features

### Chat Interface

- Clean, modern design
- Message bubbles (user vs. bot)
- Typing indicators
- Smooth animations
- Auto-scroll to latest message

### Sources Display

Sources are shown as clickable links below each answer:

📚 Sources:
- 📄 Zuora CPQ Overview
- 📄 Payment Methods Guide

### Error Handling

- Network errors
- API errors
- Backend unavailable
- Invalid responses

All errors are displayed inline in the chat.

### Status Bar

Shows current status:
- ✅ Ready
- 🔄 Thinking...
- ✅ Question answered
- ❌ Error: [message]

## Customization

### Colors

Edit CSS variables in `style.css`:

```css
:root {
    --primary-color: #2563eb;
    --user-message-bg: #2563eb;
    --bot-message-bg: #f1f5f9;
    /* ... */
}
```

### API Endpoint

Edit `app.js`:

```javascript
const API_BASE_URL = 'http://your-backend-url:8000';
```

### Welcome Message

Edit `index.html`:

```html
<div class="message bot-message">
    <div class="message-content">
        <p>Your welcome message here</p>
    </div>
</div>
```

## Browser Compatibility

- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## Responsive Design

The interface is fully responsive:

- Desktop: 900px max width, centered
- Tablet: Full width, optimized layout
- Mobile: Touch-optimized, prevents zoom

## Accessibility

- Semantic HTML
- ARIA labels (where needed)
- Keyboard navigation
- Reduced motion support
- High contrast mode compatible

## Deployment

### Deploy to Static Hosting

1. **Netlify/Vercel:**
   - Drag and drop the `frontend` folder
   - Update API_BASE_URL to production backend

2. **GitHub Pages:**
   ```bash
   # In frontend directory
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M gh-pages
   git remote add origin <your-repo>
   git push -u origin gh-pages
   ```

3. **S3/CloudFront:**
   - Upload files to S3 bucket
   - Configure CloudFront distribution
   - Update CORS in backend

### Update Backend CORS

In `backend/main.py`, add your frontend URL:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend-url.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Troubleshooting

### "Backend not reachable"

1. Check backend is running: http://localhost:8000/health
2. Check CORS settings in backend
3. Verify API_BASE_URL in app.js

### CORS Errors

Enable CORS in backend (main.py):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Sources Not Clickable

The `/ask` endpoint should return sources with `url` field:

```json
{
  "sources": [
    {
      "title": "Document Title",
      "source": "file.md",
      "url": "https://docs.zuora.com/..."  // Required for clickable links
    }
  ]
}
```

## Development

### Local Development

```bash
# Terminal 1: Start backend
cd backend
python main.py

# Terminal 2: Start frontend
cd frontend
python -m http.server 3000

# Open browser
open http://localhost:3000
```

### Live Reload

Use a tool like `live-server`:

```bash
npm install -g live-server
cd frontend
live-server --port=3000
```

## Performance

- No external dependencies
- Minimal JavaScript (~8KB)
- Fast initial load
- Efficient DOM updates
- Optimized animations

## Security

- XSS protection (HTML escaping)
- HTTPS recommended for production
- No sensitive data stored
- CORS validation required

## License

MIT License - For educational and development purposes.
