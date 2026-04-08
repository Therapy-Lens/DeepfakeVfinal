# Active Temporary Uploads Buffer

This is the primary storage area for files currently being processed by the deepfake detection engines.

### 📂 Purpose
When a user uploads a file via the frontend, the Node.js backend saves the raw file (Image, Video, or Audio) into this hidden directory. Python inference scripts then read the files from this location.

### ⚙️ Behavior
Files in this directory are **temporary**. The backend is configured to automatically delete these files shortly after the prediction result is returned to the user to maintain system performance and storage space.

### ⚠️ Note on Git
This folder is hidden (dot-prefixed) and ignored by Git. The media files processed here are extremely large and should never be committed to the repository.
