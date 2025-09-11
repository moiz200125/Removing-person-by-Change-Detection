# Change Detection Assignment

## How to Run
```bash
python3 changeDetection.py \
    --input_folder "data/person1" \
    --output_folder "results/person1" \
    --input_ext "png" \
    --output_ext "png" \
    --video_format "mp4"
Folder Structure
arduino
Copy code
RollNumber_FirstName_01/
├── changeDetection.py      # main script
├── utils/                  # helper functions
├── results/                # output masks, frames, video
├── Report.pdf
└── Readme.txt
Requirements
Python 3.8+

numpy

opencv-python

matplotlib

Install with:

bash
Copy code
pip install numpy opencv-python matplotlib
vbnet
Copy code

That’s it—short, minimal, functional. Nothing extra to annoy the grader.  

Want me to also trim it down to a **single Readme.txt style** (just run command + requirements, no folder tree)?