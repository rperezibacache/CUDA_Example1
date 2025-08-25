# INSTALL PROCESS GPU
- First, Check Your Ubuntu Version
```bash
lsb_release -a
```
- Add the Official NVIDIA Repository
```bash
# Install prerequisites
sudo apt update
sudo apt install software-properties-common
```
## Add the graphics drivers PPA
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
```
- Install the Appropriate Driver for RTX 4070
Your RTX 4070 requires driver version 525 or newer. Let's install the latest:

```bash
# Check available driver versions
ubuntu-drivers devices

# Install the latest recommended driver (should be 535 or newer)
sudo apt install nvidia-driver-535

# Alternatively, if you want the very latest
sudo apt install nvidia-driver-545
```

- Check driver
```bash
# After reboot, check if driver works
nvidia-smi
```
# INSTALL CUDA 
- Update package lists
```bash
sudo apt update
```
- Install CUDA 12.2 toolkit ONLY (no driver)
```bash
sudo apt install cuda-toolkit-12-2
```
- If that doesn't work, try the meta-package
```bash
sudo apt install cuda-toolkit-12-*
```
- Or install from NVIDIA repo
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install cuda-toolkit-12-2
```
- Completely reset your environment
```bash
echo '' > ~/.bashrc  # Backup your .bashrc first if needed!
```
- Add only the essential CUDA paths
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```
- Apply
```bash
source ~/.bashrc
```
- Check what nvcc points to
```bash
which nvcc
readlink -f $(which nvcc)
```
- Check CUDA version
```bash
nvcc --version
```
- Check the actual CUDA directory
```bash
ls -la /usr/local/cuda
cat /usr/local/cuda/version.txt 2>/dev/null
```
- Check library paths
```bash
echo $LD_LIBRARY_PATH
```

# run example 
```bash
nvcc gpu_test.cu -o gpu_test
./gpu_test
```
- Expected return
```bash
Number of CUDA devices: 1
Device 0: NVIDIA GeForce RTX 4080
  Compute capability: 8.9
  Total GPU memory: 15.69 GB
```
# Useful commands
- Monitor GPU usage while your program runs
```bash
watch -n 0.1 nvidia-smi
```

