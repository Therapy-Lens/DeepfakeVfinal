# Backend Dependency Documentation

This directory contains the Node.js server logic for handling file uploads and triggering the machine learning pipeline.

### 📦 Externally Installed Dependencies

The following packages are installed via npm and are essential for the backend's operation:

| Package Name | Purpose | Installation Command |
| :--- | :--- | :--- |
| **express** | Core web framework for the API | `npm install express` |
| **cors** | Enables Cross-Origin Resource Sharing for the frontend | `npm install cors` |
| **multer** | Middleware for handling multipart/form-data (file uploads) | `npm install multer` |
| **axios** | For potential external API calls or routing | `npm install axios` |
| **form-data** | Library to create readable "multipart/form-data" streams | `npm install form-data` |

### 🚀 Setup Instructions

To install these dependencies manually in a fresh environment, navigate to this directory and run:

```bash
npm install express cors multer axios form-data
```

Or simply run `npm install` to install everything listed in the `package.json`.
