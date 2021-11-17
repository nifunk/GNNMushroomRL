# create folder for additional dependencies
mkdir additional_dependencies
cd additional_dependencies
git clone https://github.com/famura/SimuRLacra.git
cd SimuRLacra
git checkout 1812a53cd900254fb0fc6c1f565bd71231199ab1
cd Pyrado
pip install -e .
