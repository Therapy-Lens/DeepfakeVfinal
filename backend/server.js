const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const uploadRoute = require("./routes/upload");

const uploadPath = path.join(__dirname, ".uploads");

if (!fs.existsSync(uploadPath)) {
  fs.mkdirSync(uploadPath);
}

const app = express();

app.use(cors());
app.use(express.json());

app.use("/", uploadRoute);

app.listen(5000, () => {
  console.log("Server running on port 5000");
});
