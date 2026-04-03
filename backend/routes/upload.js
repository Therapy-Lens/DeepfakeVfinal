const express = require("express");
const multer = require("multer");
const path = require("path");

const router = express.Router();

// storage config
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, "../uploads"));
  },
  filename: (req, file, cb) => {
    const uniqueName = Date.now() + path.extname(file.originalname);
    cb(null, uniqueName);
  },
});

const upload = multer({ storage });

const { execFile } = require("child_process");
const fs = require("fs");

router.post("/upload", upload.array("files", 10), async (req, res) => {
  try {
    const files = req.files;

    if (!files || files.length === 0) {
      return res.status(400).json({ error: "No files uploaded" });
    }

    console.log("Files received:", files.length);

    const filePath = files[0].path;
    const isVideo = files[0].mimetype.startsWith('video');
    const isAudio = files[0].mimetype.startsWith('audio');
    
    // SAFE GUARD FOR AUDIO FILES
    if (isAudio) {
      console.log("Audio file received natively. Sending placeholder response.");
      res.json({ "prediction": "AUDIO RECEIVED", "confidence": 0 });
      
      // Cleanup locally exactly mimicking organic explicit flows natively
      files.forEach(f => {
          try {
              if (fs.existsSync(f.path)) fs.unlinkSync(f.path);
          } catch (err) { console.error("File delete error:", err); }
      });
      return;
    }

    const pythonPath = path.join(__dirname, "../../venv/Scripts/python.exe");
    const scriptPath = isVideo 
      ? path.join(__dirname, "../utils/predict_video.py")
      : path.join(__dirname, "../utils/predict.py");

    const { exec } = require("child_process");
    
    // Command wraps paths in double quotes in case of spaces in directories
    const command = `"${pythonPath}" "${scriptPath}" "${filePath}"`;
    
    exec(command, (err, stdout, stderr) => {
      console.log("Python script finished execution.");
      if (stdout) console.log("STDOUT:", stdout);
      if (stderr) console.log("STDERR:", stderr);

      if (err) {
        console.error("Execution Error:", err.message);
        // Do not crash server, just tell frontend it failed
        return res.status(500).json({ error: "Prediction failed" });
      }

      try {
        console.log("Final stdout:", stdout);

        const lines = stdout.trim().split('\n');
        const lastLine = lines[lines.length - 1];

        const result = JSON.parse(lastLine);
        console.log("Parsed JSON:", result);
        
        res.json(result);
        
        // AFTER response
        files.forEach(f => {
            try {
                if (fs.existsSync(f.path)) {
                    fs.unlinkSync(f.path)
                }
            } catch (err) {
                console.error("File delete error:", err)
            }
        });
        
      } catch (parseError) {
        console.error("JSON parse error on string:", stdout);
        res.status(500).json({ error: "Invalid prediction output" });
      }
    });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Upload failed" });
  }
});

module.exports = router;
