#! /bin/sh

# Install polymetis from source
git clone git@github.com:RUreadyo/monometis.git

cd monometis/
mamba env create -f polymetis/environment_cpu.yml -n manimo-latest
conda activate manimo-latest

# compile stuff, no need to build libfranka on this machine
./scripts/build_libfranka.sh
mkdir -p ./polymetis/build
cd ./polymetis/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=ON
make -j
cd ../..

pip install -e ./polymetis
