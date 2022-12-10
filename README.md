This is the pytorch implementation of the paper "Deep Gradient Learning for Efficient Camouflaged Object Detection" by Ge-Peng Ji, Deng-Ping Fan, Yu-Cheng Chou, Dengxin Dai, Alexander Liniger, Luc Van Gool. The paper can be found [here](https://doi.org/10.48550/arXiv.2205.12853).

This is modified to work with 2 new datasets:

* Military Camouflage Dataset (MCD)
* MoCA Video Dataset

# How to run test.py

just run `python test.py` and it will run the test on the MoCA dataset. You can change the dataset by changing the `dataset` variable in the `test.py` file. Uncomment the code for different datasets.

You can get the data set from the links provided in our HTML Report.

[Here](https://utoronto-my.sharepoint.com/:u:/g/personal/trupal_patel_mail_utoronto_ca/EW2tAtrqzJNDvx_DWw_oGMABGI_PJ8fLlXFJcWazxi8i3Q?e=cIa8E0) is the link of the snapshot of the best model.


`masking.py`:
   - creates mask for MoCA dataset and makes a video of the masked images

