# Use a specific version of Ubuntu for stability
FROM ubuntu:20.04

# Set working directory
WORKDIR /usr/app/src

# Set environment variables
ENV LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8"\
    DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    locales \
    python3-pip \
    python3-yaml \
    rsyslog \
    systemd \
    systemd-cron \
    sudo \
    build-essential \
    cmake \
    libboost-all-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python packages
COPY LP4_STAPP_Requirements.txt ./
RUN pip3 install --upgrade pip && pip3 install -r LP4_STAPP_Requirements.txt


# Copy the rest of the application code
COPY ./ ./

# Specify the command to run the application
CMD [ "streamlit", "run", "Churn_Prediction.py" ]