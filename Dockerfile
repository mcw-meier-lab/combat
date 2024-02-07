FROM rocker/r2u:jammy

ENV DASH_DEBUG_MODE True
ENV PATH="/usr/local/lib/R/site-library:${PATH}"
ARG PATH="/usr/local/lib/R/site-library:${PATH}"

COPY ./app.py /app.py
COPY ./enigma.R /enigma.R

RUN apt-get update
RUN apt-get install -y software-properties-common \
    build-essential \
    bzip2 \
    libassuan-dev \
    libgcrypt20-dev \
    libgpg-error-dev \
    libksba-dev \
    libnpth0-dev \
    gnupg
RUN add-apt-repository universe
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.9
RUN apt-get install -y python3-pip \
    libpng-dev \
    libpython3-dev \
    python3.9-dev \
    python3.9-distutils \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN /usr/bin/python3.9 -m pip install --upgrade setuptools
RUN /usr/bin/python3.9 -m pip install dash_bio \
    neuroHarmonize \
    matplotlib \
    dash_bootstrap_components \
    numpy \
    pandas \
    plotly==5.13.1 \
    scipy==1.11.4 \
    formulaic \
    rpy2==3.5.9 \
    statsmodels \
    nibabel \
    pyarrow \
    dash_daq

RUN install2.r \
    BiocManager \
    sva \
    combat.enigma \
    devtools \
    lme4 \
    pbkrtest

RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN installGithub.r jfortin1/neuroCombat_Rpackage jcbeer/longCombat

EXPOSE 8050
ENTRYPOINT ["/usr/bin/python3.9", "/app.py"]