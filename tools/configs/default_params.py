# A single real video (or image) to seed one simulation from, used by
# generate_synthetic_video.py's main(). Pick any frame interactively
# instead by calling run_simulation() directly, as the notebook does.
INPUT_VIDEO = r"/path/to/a/real/cell/video.mp4"
# Where that simulation's (img_t/img_tp1/flow/sim.mp4) files get written.
OUTPUT_DIR = r"/path/to/output"
# Fixes which random frame gets picked and the random motion field, for a
# reproducible simulation. None = different every run.
SEED = 0

# Cell segmentation method: "sam" (SAM 2, most accurate, needs a checkpoint
# + torch) or "hough" (a plain Hough circle as the mask, no extra
# dependencies). SAM 2 is itself seeded by a Hough circle either way --
# see segmentation.py.
SEGMENTATION_METHOD = "sam"
# Path to the SAM 2 checkpoint (.pt) -- only used when SEGMENTATION_METHOD = "sam".
SAM_CHECKPOINT = r"/path/to/sam2.1_hiera_small.pt"
# SAM 2 model config (Hydra yaml) -- ships inside the sam2 package itself.
SAM_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
# "cuda" or "cpu" for SAM 2 inference.
SAM_DEVICE = "cuda"

# Resize the seed frame to (height, width) before segmentation/simulation,
# so every generated video (and its ground-truth flow) is natively this
# size -- no rescaling afterwards. None = keep the frame's original size.
TARGET_SIZE = (256, 256)

# Number of (img_t, img_tp1, flow) triplets to generate per simulation.
FRAMES_PER_SIM = 200

# Weight of the density-gradient-following component of the velocity field
# (0 = none). The real dataset used 0.0 -- purely random/turbulent motion.
GRAD_WEIGHT = 0.0
# Weight of the random (turbulent) component.
RAND_WEIGHT = 1.0
# Overall velocity field magnitude after combining the two components above.
INTENSITY = 0.35
# Gaussian smoothing (sigma, px) applied to the combined field -- higher is
# smoother/more viscous motion, lower is noisier/more granular.
VISCOSITY = 4.0
# Strength (0-1) of the membrane non-penetration correction at the cell
# boundary: 0 = no correction at all (density can cross the mask edge
# freely), 1 = the paper's Eq. 10 exactly (full-strength projection) --
# which visibly bakes the mask's contour into the video as a ring over many
# frames. 0.4 is a deliberate, softened middle ground -- see physics.py's
# module docstring for the full story.
BOUNDARY_DAMPING = 0.4

# Output video.
MP4_FPS = 20
MP4_FOURCC = "mp4v"
