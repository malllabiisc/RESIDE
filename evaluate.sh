sh setup.sh

# Evaluate the pre-trained model
python reside.py -data data/riedel_processed.pkl -name pretrained_model -restore -only_eval

# Plot PR-curve
python plot_pr.py -name pretrained_riedel -dataset riedel