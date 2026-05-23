import matplotlib.pyplot as plt
import xgboost as xgb

# 1. Create an empty model object
model = xgb.XGBRegressor()

# 2. Set the simple path to your model file
model_path = "../../models/xgboost_cost_model.json"

# 3. Load the model file
model.load_model(model_path)
print("Successfully loaded: ../../models/xgboost_cost_model.json")

# 4. Create the chart layout with a highly balanced figure window
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

# 5. Plot the feature importance chart (We pass title="" here to clear the broken title)
xgb.plot_importance(
    model, 
    max_num_features=10, 
    importance_type='weight', 
    title='', # Clear default title
    xlabel='F-Score (Number of times feature was split)', 
    ylabel='Extracted Pipeline Features',
    color='#2563eb', 
    ax=ax,
    grid=False
)

# 6. Apply highly specific, cleaner text styling properties
# FIX: Using fig.suptitle forces the title downwards and centers it perfectly inside the window
fig.suptitle('XGBoost Cost Engine - Feature Importance', fontsize=12, fontweight='bold', y=0.96)

ax.set_xlabel('F-Score (Number of times feature was split)', fontsize=9, labelpad=10)
ax.set_ylabel('Extracted Pipeline Features', fontsize=9, labelpad=10)

# Shrink the font sizes of the feature names and axis values so they fit comfortably
ax.tick_params(axis='both', which='major', labelsize=9)

# Loop through the value labels on the bars to shrink their font sizes too
for text in ax.texts:
    text.set_size(8.5)

# 7. Add a subtle background grid line
ax.grid(True, linestyle='--', alpha=0.5, color='#ccc')

# Balanced margins to give breathing room to left text, right numbers, and the new title
plt.subplots_adjust(left=0.28, right=0.92, top=0.86, bottom=0.15)

# 8. Save the chart as an image file
plt.savefig("feature_importance.png", bbox_inches='tight')
print("Success! Perfect 'feature_importance.png' has been saved inside src/cost/")
plt.show()