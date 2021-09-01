

# copy agent file and model
ln -s ../src/Agent/sac_agent.py agent.py
cp ../src/sac_trial.zip model.zip

# copy other used files

# zip submission 
zip -r submission.zip submission

# upload submission.zip on the website

