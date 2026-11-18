# exec the notebook cells that build df/T/X_all and the mixed-halide tables, without the scorecard
import json, re
nb = json.load(open("descriptor_selection.ipynb"))
src = lambda i: "".join(nb["cells"][i]["source"])
g = globals()
for i in [1, 3, 5, 7, 9]:
    exec(src(i).replace('plt.style.use("my_style")', 'pass'), g)
exec(src(17).split("\n\ndef cv_r2")[0], g)  # model()
exec(src(27), g)
exec(src(28).split("\n\nrows = {}")[0], g)  # FEATS, r2_vs, fit_predict
picked = list(pd.read_csv(OUT_DIR / "descriptor_scorecard.csv", index_col=0).query("shortlist").index)
