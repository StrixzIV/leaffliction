.PHONY: lint download norm

images:
	wget https://cdn.intra.42.fr/document/document/39704/leaves.zip
	unzip leaves.zip
	mkdir -p images/Apple images/Grape
	mv images/Apple_* images/Apple
	mv images/Grape_* images/Grape

download: images

norm:
	flake8 .
