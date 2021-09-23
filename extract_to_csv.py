import os
import utils


codedir = "/Users/admin/UDel/FASTLab/Summer2021_Research/SESNspectraPCA/code"
datadir = os.path.join(codedir, "../Data/DataProducts")
savedir = os.path.join(codedir, "../wfortino/csv_data")

pickle0 = os.path.join(datadir, "dataset0.pickle")
pickle5 = os.path.join(datadir, "dataset5.pickle")
pickle10 = os.path.join(datadir, "dataset10.pickle")
pickle15 = os.path.join(datadir, "dataset15.pickle")

utils.dataset_to_csv(pickle0, datadir, os.path.join(savedir, "dataset0"))
utils.dataset_to_csv(pickle5, datadir, os.path.join(savedir, "dataset5"))
utils.dataset_to_csv(pickle10, datadir, os.path.join(savedir, "dataset10"))
utils.dataset_to_csv(pickle15, datadir, os.path.join(savedir, "dataset15"))
