FROM jenkins/jenkins:lts


#switch to root user to install docker
 USER root

 # Install Docker CLI
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create the docker group and add jenkins user to it
RUN groupadd docker || true && \
    usermod -aG docker jenkins

# Create the docker directory and volume
RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker
# Switch back to the jenkins user
    USER jenkins