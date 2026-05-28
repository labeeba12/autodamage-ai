import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb

# Create an empty model object
model = xgb.XGBRegressor()

# Resolve the model path robustly (script can be run from different CWDs)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(script_dir, '..', '..', 'models', 'xgboost_cost_model.json'))

# Fallback: try project-root-relative path as a secondary option
alt_model_path = os.path.normpath(os.path.join(os.getcwd(), 'models', 'xgboost_cost_model.json'))

if not os.path.exists(model_path) and os.path.exists(alt_model_path):
    model_path = alt_model_path

if not os.path.exists(model_path):
    print(f"Model file not found. Tried:\n  {model_path}\n  {alt_model_path}\nPlease ensure the file exists and retry.")
    raise SystemExit(1)

try:
    model.load_model(model_path)
    print(f"Successfully loaded: {model_path}")
except Exception as e:
    print(f"Failed to load model at {model_path}: {e}")
    raise

# Create the chart layout with extra horizontal room so labels and numbers fit
fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

# Plot the feature importance chart (we'll draw value labels manually to control placement)
xgb.plot_importance(
    model,
    max_num_features=10,
    importance_type='weight',
    title='',  # Clear default title
    xlabel='F-Score (Number of times feature was split)',
    ylabel='Extracted Pipeline Features',
    color='#2563eb',
    ax=ax,
    grid=False
)

# Title and axis labels
fig.suptitle('XGBoost Cost Engine - Feature Importance', fontsize=14, fontweight='bold', y=0.96)
ax.set_xlabel('F-Score (Number of times feature was split)', fontsize=10, labelpad=10)
ax.set_ylabel('Extracted Pipeline Features', fontsize=10, labelpad=10)

# Tidy tick sizes
ax.tick_params(axis='both', which='major', labelsize=10)

# Remove any automatic value texts added by xgboost and draw our own to ensure visibility
for t in list(ax.texts):
    t.set_visible(False)

# Compute a small offset for the numeric labels so they sit just to the right of the bar
widths = [p.get_width() for p in ax.patches]
max_w = max(widths) if widths else 0
offset = max_w * 0.01 if max_w else 1.0

for p in ax.patches:
    w = p.get_width()
    y = p.get_y() + p.get_height() / 2.0
    label = f"{w:.0f}" if w.is_integer() else f"{w:.1f}"
    # place label slightly to the right, ensure it's vertically centered and not clipped
    ax.text(w + offset, y, label, va='center', fontsize=9, color='#111111', clip_on=False)

# Remove the external legend (title already describes the chart) and use a short caption
# Subtle grid for readability
ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')

# Ensure there's plenty of room at left and right for labels and annotation numbers
# increase bottom so x-axis label and ticks are fully visible
plt.subplots_adjust(left=0.28, right=0.98, top=0.86, bottom=0.20)

# Save the chart as an image file
plt.savefig("feature_importance.png", bbox_inches='tight')
print("Saved feature_importance.png inside src/cost/")
plt.show()