sudo apt-get update && sudo apt-get install -y libsndfile2

if conda env list | grep -q voice_cloning; then
    echo voice_cloning environment already exists!
else
    echo creating new environment
    conda env create -f environment.yml
fi

echo activating voice_cloning environment...
conda activate voice_cloning
echo done!
