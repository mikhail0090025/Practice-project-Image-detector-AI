pipeline {
    agent any
    options {
        // Timeout counter starts AFTER agent is allocated
        timeout(time: 1, unit: 'SECONDS')
    }
    stages {
        stage('First step') {
            steps {
                echo 'This is first step'
            }
        }
        stage('Build Docker Images') {
            steps {
                sh '''
                    docker-compose build
                '''
            }
        }
        stage('Run Containers') {
            steps {
                sh '''
                    docker-compose up -d
                '''
            }
        }
        stage('Save Logs') {
            steps {
                sh '''
                    docker-compose logs neural_net > neural_net.log
                '''
                archiveArtifacts artifacts: 'neural_net.log', allowEmptyArchive: true
            }
        }
    }
    post {
        always {
            sh '''
                docker-compose down
            '''
            echo 'Pipeline finished!'
        }
        success {
            echo 'GAN training completed successfully!'
        }
        failure {
            echo 'GAN training failed! Check logs.'
        }
    }
}