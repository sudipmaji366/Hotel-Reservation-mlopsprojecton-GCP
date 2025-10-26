pipeline{
    agent any
        environment {
            VENV_DIR='venv'
            
        }
    stages{
        stage('clone repository'){
            steps{
                script {

                    echo "Cloning the repository"
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'GitHub-token', url: 'https://github.com/sudipmaji366/Hotel-Reservation-mlopsprojecton-GCP.git']])
                }
        }
    }
    stage('Setting up virtual environment and installing dependencies'){
            steps{
                script {

                    echo "Setting up virtual environment and installing dependencies"
                    sh '''

                        python3 -m venv $VENV_DIR
                        . $VENV_DIR/bin/activate
                        pip install --upgrade pip
                        pip install -e .
                    '''
        }
    }
}
}