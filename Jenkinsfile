pipeline{
    agent any

    stages{
        stage('clone repository'){
            steps{
                script {

                    echo "Cloning the repository"
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'GitHub-token', url: 'https://github.com/sudipmaji366/Hotel-Reservation-mlopsprojecton-GCP.git']])
                }
        }
    }
}
}