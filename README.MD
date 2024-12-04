# 投影片擷取器 / Slide Extractor

## 說明 / Description
### 系統需求 / System Requirements
- OpenCV 4.x
- CMake 3.10+
- C++17 編譯器

### 建置步驟 / Build Instructions
```bash
mkdir build && cd build
cmake ..
make
```

### 使用方法 / Usage
#### 中文
```bash
./slide_extractor <影片路徑> <輸出目錄> [開始畫格] [結束畫格]
```
#### English
```bash
./slide_extractor <video_path> <output_directory> [start_frame] [end_frame]
```

### 參數說明
- `<影片路徑>` / `<video_path>`：輸入影片檔案路徑 / Input video file path
- `<輸出目錄>` / `<output_directory>`：儲存擷取投影片的資料夾 / Directory to save extracted slides
- `[開始畫格]` / `[start_frame]`：可選，起始畫格（預設：0） / Optional starting frame (default: 0)
- `[結束畫格]` / `[end_frame]`：可選，結束畫格（預設：整段影片） / Optional ending frame (default: entire video)

### 功能特點/Features
- 偵測投影片變換 / Detect slide changes
- 擷取最大白色區域 / Extract largest white region
- 儲存編號 PNG 圖檔 / Save numbered PNG files


### 輸出格式 / Output Filename Format
`result_slideNo##_frameNo##.png`

## 疑難排解 / Troubleshooting
- 確認已正確安裝 OpenCV / Ensure OpenCV is correctly installed
- 檢查執行檔權限 / Check executable permissions
- 驗證輸入影片檔案路徑 / Verify input video file path

## 範例 / Example
```bash
./slide_extractor presentation.mp4 slides_output
```