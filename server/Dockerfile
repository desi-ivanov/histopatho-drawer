FROM nvcr.io/nvidia/pytorch:20.12-py3
WORKDIR /workspace
COPY . .
RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]
