# Zuora Help Agent - Currently Running

## ✅ Services Status

### Backend API
- **URL**: http://localhost:8000
- **Status**: ✅ Running (PID: check with `ps aux | grep uvicorn`)
- **Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **Logs**: `logs/backend.log`

### Frontend
- **URL**: http://localhost:3001
- **Status**: ✅ Running
- **Logs**: `logs/frontend.log`

## 🧪 Quick Tests

### Test Backend Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development"
}
```

### Test Ask Endpoint
```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is Zuora CPQ?"}'
```

Expected response:
```json
{
  "answer": "This endpoint will be implemented with RAG capabilities. Your question was: What is Zuora CPQ?",
  "sources": ["placeholder_source.pdf"],
  "conversation_id": "new_conversation_id",
  "confidence": 0.0
}
```

### Test Frontend
Open in browser: http://localhost:3001

## 📝 Usage

### Using the Chat Interface

1. **Open Frontend**: http://localhost:3001
2. **Type a question**: e.g., "What is Zuora CPQ?"
3. **Click Send** or press Enter
4. **View response**: The bot will respond with an answer

### Example Questions

Try these in the chat interface:
- "What is Zuora CPQ?"
- "How do I configure SSO?"
- "What payment methods does Zuora support?"
- "Tell me about subscription management"

## 🛑 Stop Services

### Stop Backend
```bash
pkill -f "uvicorn main:app"
```

### Stop Frontend
```bash
pkill -f "http.server 3001"
```

### Stop All
```bash
pkill -f "uvicorn main:app"
pkill -f "http.server 3001"
```

## 🔄 Restart Services

### Restart Backend
```bash
# Stop
pkill -f "uvicorn main:app"

# Start
cd backend
/Users/bharathmarimuthu/Library/Python/3.9/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > ../logs/backend.log 2>&1 &
```

### Restart Frontend
```bash
# Stop
pkill -f "http.server 3001"

# Start
cd frontend
python3 -m http.server 3001 > ../logs/frontend.log 2>&1 &
```

## 📊 View Logs

### Backend Logs (Live)
```bash
tail -f logs/backend.log
```

### Frontend Logs (Live)
```bash
tail -f logs/frontend.log
```

### Recent Backend Activity
```bash
tail -20 logs/backend.log
```

## 🐛 Troubleshooting

### Port Already in Use

If you get "Address already in use" error:

```bash
# Find process using the port
lsof -ti:8000  # Backend
lsof -ti:3001  # Frontend

# Kill the process
kill -9 $(lsof -ti:8000)
kill -9 $(lsof -ti:3001)
```

### Backend Not Responding

1. Check if running: `ps aux | grep uvicorn`
2. Check logs: `tail -20 logs/backend.log`
3. Restart the backend (see above)

### Frontend Not Loading

1. Check if running: `ps aux | grep 'http.server 3001'`
2. Check logs: `tail -20 logs/frontend.log`
3. Try accessing: http://localhost:3001
4. Clear browser cache and reload

### CORS Errors in Browser

The backend is configured with CORS middleware. If you see CORS errors:

1. Verify backend is running on port 8000
2. Check frontend is accessing http://localhost:8000
3. Check browser console for specific error

## 📈 Next Steps

### To Implement RAG Pipeline:

1. **Scrape Documentation**
   ```bash
   cd scripts
   python scrape_zuora_sitemap.py --product billing --max-pages 50
   ```

2. **Build FAISS Index**
   ```bash
   export VOYAGE_API_KEY="your_key"
   python build_index.py --input ../data/zuora_docs.json
   ```

3. **Update Backend**
   - Implement RAG logic in `backend/rag.py`
   - Update `/ask` endpoint in `backend/main.py`
   - Use `faiss_loader.py` to load the index

4. **Test with Real Data**
   - Ask questions about Zuora products
   - Verify source attribution
   - Check response quality

## 🌐 Access URLs

- **Frontend**: http://localhost:3001
- **Backend Health**: http://localhost:8000/health
- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc

## 💡 Tips

- **Keep both terminals open** to see logs in real-time
- **Use browser DevTools** to debug frontend issues
- **Check API docs** at /docs for request/response formats
- **Monitor logs** when testing to see what's happening

## 📱 Mobile Testing

The frontend is responsive! Test on mobile:
1. Find your local IP: `ifconfig | grep "inet "`
2. Access from phone: http://YOUR_IP:3001

---

**Status**: Both services are running and ready for testing!

Last updated: 2026-02-27
