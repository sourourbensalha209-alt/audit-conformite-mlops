# Image de base du Codespace.
# L'image Python de Microsoft déclare un dépôt Yarn dont la clé de signature n'est plus valide (NO_PUBKEY 62D54FD4003F6525) :
# « apt-get update » échoue alors pendant l'installation de Docker. Yarn est inutile ici : on retire ce dépôt.
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm
RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn*.list
