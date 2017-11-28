# Basic VAE Example

This is an improved implementation of the paper [Stochastic Gradient VB and the
Variational Auto-Encoder](http://arxiv.org/abs/1312.6114) by Kingma and Welling.
It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster.

```bash
pip install -r requirements.txt
python main.py
```

# Setup | For MILA people

<<In the following instructions, replace <yourname> with “sai” >>

```Git instructions

git clone https://github.com/nithin127/nest-vae
cd nest-vae
git checkout -b devel-sai 	#to create your own branch named “devel-sai”
git add *			#to add all the files to be committed
git commit -m "comments"	#to comment
git push			#it will prompt you to type the next command
git push --set-upstream origin devel-yourname

To get the tensorboard working:

conda create --name sai python=3.6	#just do it. Don’t question 
source activate sai
conda install pytorch torchvision cuda80 -c soumith
pip install tensorflow-gpu
pip install tensorboardX
pip uninstall torchvision 		#coz existing version is crooked
pip install git+https://github.com/pytorch/vision.git	#This is the correct one


cd nest-vae
git checkout devel-tristan	#to go into tristan’s directory. Yes, “devel-tristan” NOT “devel-yourname”
git branch			#idk why, but pls do this
git pull				#to pull all his recent commits
cd tristan/
cd pytorch_tutorial_vae/	#this is where you will get enlightened

<<In a different terminal>>
<<replace gottipav with your elisaID in the following instructions>>
<<replace 1996 with your year of birth>>
ssh -X gottipav@elisa1.iro.umontreal.ca -L 6006:localhost:1996
ssh -X gottipav@bart15.iro.umontreal.ca -L 1996:localhost:6006 		#bart15 is the GPU. 
replace it with whatever GPU you are using
Activate your environment
Go to your directory
tensorboard --logdir .logs --port 6006

Now, python main.py in the /nest-vae/tristan/pytorch_tutorial_vae in your earlier terminal and you can see the tensorboard opening in a browser and doing some stuff

python main.py

```

