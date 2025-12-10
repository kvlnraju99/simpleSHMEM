# NVSHMEM Installation Methods

## 🚨 Problem

Direct downloads from NVIDIA's servers are failing with 404 errors. This is likely due to:

- **Authentication requirements** on NVIDIA's download server
- **Network restrictions** on the HPC cluster
- **Firewall/proxy issues**

## ✅ Solution: 3 Installation Methods

Choose the method that works best for your HPC cluster:

---

## Method 1: APT Package Manager (RECOMMENDED)

**Best for**: Ubuntu/Debian-based HPC clusters with sudo access

```bash
bash install_nvshmem_apt.sh
```

### What it does:

1. Adds NVIDIA CUDA repository
2. Installs `libnvshmem3-cuda-12` and `libnvshmem3-dev-cuda-12`
3. Auto-detects installation path
4. Sets environment variables

### Advantages:

- ✅ No manual downloads
- ✅ Automatic dependency resolution
- ✅ Easy updates via `apt upgrade`

---

## Method 2: Manual Download (Local Machine → HPC)

**Best for**: When HPC cluster blocks downloads

### Steps:

#### On your local machine:

```bash
# Download the file (this might work on your local machine with different network)
wget https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz

# Or use curl
curl -O https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz
```

#### Transfer to HPC:

```bash
# Use scp to transfer
scp libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz vk2646@hpc-cluster:~/software/
```

#### On HPC cluster:

```bash
cd ~/software
tar -xf libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz
export NVSHMEM_HOME=~/software/libnvshmem-linux-x86_64-3.4.5_cuda12-archive
echo "export NVSHMEM_HOME=~/software/libnvshmem-linux-x86_64-3.4.5_cuda12-archive" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$NVSHMEM_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

---

## Method 3: Use Older Version (Fallback)

**Best for**: When latest version is unavailable

Try an older stable version (3.2.5):

```bash
cd ~/software
wget https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/libnvshmem-linux-x86_64-3.2.5_cuda12-archive.tar.xz
tar -xf libnvshmem-linux-x86_64-3.2.5_cuda12-archive.tar.xz
export NVSHMEM_HOME=~/software/libnvshmem-linux-x86_64-3.2.5_cuda12-archive
```

---

## Method 4: Check if Already Installed

**Best for**: Enterprise HPC clusters

Many HPC clusters pre-install NVSHMEM via modules:

```bash
# Search for NVSHMEM module
module avail nvshmem

# If found, load it
module load nvshmem/3.x-cuda-12

# Check environment
echo $NVSHMEM_HOME
```

---

## Verification

After installation (any method), verify:

```bash
# Check files exist
ls -la $NVSHMEM_HOME/include/nvshmem.h
ls -la $NVSHMEM_HOME/lib/libnvshmem_host.so

# Compile your code
cd ~/simpleSHMEM
make clean && make

# Should compile without errors
```

---

## Troubleshooting

| Error                        | Solution                                         |
| ---------------------------- | ------------------------------------------------ |
| `404 Not Found`              | Try Method 1 (APT) or Method 2 (manual transfer) |
| `Permission denied`          | Add `sudo` or contact HPC admin                  |
| `module: command not found`  | HPC doesn't use modules, try APT method          |
| `cannot find -lnvshmem_host` | Run `source ~/.bashrc` or logout/login           |

---

## Which Method Should I Use?

1. **Do you have sudo access?** → Try Method 1 (APT)
2. **Is wget blocked?** → Try Method 2 (manual download + transfer)
3. **On enterprise HPC?** → Try Method 4 (check modules)
4. **All else fails?** → Try Method 3 (older version)
