# IPC Interactive Lecture (Streamlit)

This repo contains a single Streamlit app with interactive activities for an IPC/RPC lecture.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repo.
2. Add these files:
   - `streamlit_app.py`
   - `requirements.txt`
3. On https://share.streamlit.io, create a new app and point to `streamlit_app.py` as the **Main file path**.
4. If the app hangs at *Provisioning*, try:
   - Make sure `requirements.txt` exists (pin `streamlit>=1.30`).
   - Ensure the main file is named `streamlit_app.py` or set correctly in app settings.
   - Clear/reboot the app (in the app's **Settings → Manage app → Restart**).
   - If Python version mismatch occurs, add a `runtime.txt` with `3.11`.

## Notes
- The app uses only the standard library plus Streamlit; no network access or secrets required.
- If you previously used `st.tags_input`, this file already uses simple text inputs for maximum compatibility.
