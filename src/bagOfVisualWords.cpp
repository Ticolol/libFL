//
// Created by deangeli on 4/7/17.
//

#include "bagOfVisualWords.h"
#include "math.h"

#define DICT_DIFFERENCE_EPSILON 9e-7


FeatureMatrix* computeFeatureVectors(DirectoryManager* directoryManager, int patchSize){
    Image* currentSlice;
    Image* patch;
    Histogram* histogram;
    FeatureVector* patchVector;
    Image* firstImage = readImage(directoryManager->files[0]->path);
    int patchX_axis = firstImage->nx/patchSize;
    int patchY_axis = firstImage->ny/patchSize;
    int numberPatchsPerImage = patchX_axis*patchY_axis;
    int numberPatchs = numberPatchsPerImage*directoryManager->nfiles;
    int binSize = 64;
    destroyImage(&firstImage);
    FeatureMatrix* featureMatrix = createFeatureMatrix(numberPatchs);
    int k=0;
    for (size_t fileIndex = 0; fileIndex < directoryManager->nfiles; ++fileIndex) {
        Image* currentImage = readImage(directoryManager->files[fileIndex]->path);

        #pragma omp parallel for
        for (int z = 0; z < currentImage->nz; ++z) {
            currentSlice = getSlice(currentImage,z);
            for (int y = 0; y <= currentImage->ny-patchSize; y +=patchSize) {
                for (int x = 0; x <= currentImage->nx-patchSize; x += patchSize) {
                    patch = extractSubImage(currentSlice,x,y,patchSize,patchSize,true);
                    histogram = computeHistogram(patch,binSize,true);
                    patchVector = createFeatureVector(histogram);
                    featureMatrix->featureVector[k] = patchVector;
                    k++;
                    destroyHistogram(&histogram);
                    destroyImage(&patch);
                }
            }
            destroyImage(&currentSlice);
        }

        destroyImage(&currentImage);
    }
    return featureMatrix;
}


FeatureMatrix* computeFeatureVectors(Image* imagePack, int patchSize)
{
    Image* currentSlice;
    Image* patch;
    Histogram* histogram;
    FeatureVector* patchVector;
    int patchX_axis = imagePack->nx/patchSize;
    int patchY_axis = imagePack->ny/patchSize;
    int numberPatchsPerImage = patchX_axis*patchY_axis;
    int numberPatchs = numberPatchsPerImage*imagePack->nz;
    int binSize = 64;

    FeatureMatrix* featureMatrix = createFeatureMatrix(numberPatchs);
    int k=0;
    for (int z = 0; z < imagePack->nz; ++z) {
        currentSlice = getSlice(imagePack,z);
        for (int y = 0; y <= imagePack->ny-patchSize; y +=patchSize) {
            for (int x = 0; x <= imagePack->nx-patchSize; x += patchSize) {
                patch = extractSubImage(currentSlice,x,y,patchSize,patchSize,true);
                histogram = computeHistogram(patch,binSize,true);
                patchVector = createFeatureVector(histogram);
                featureMatrix->featureVector[k] = patchVector;
                k++;
                destroyHistogram(&histogram);
                destroyImage(&patch);
            }
        }
        destroyImage(&currentSlice);
    }
    return featureMatrix;
}

float euclidean_distance(FeatureVector *v0, FeatureVector *v1)
{
	int length;
	float distance = 0;
    length = v0->size;
	if (length != v1->size) {
		return NAN;
	}
	for (int i = 0; i < length; i++) {
		float difference = v0->features[i] - v1->features[i];
		distance += difference * difference;
	}
	distance = sqrt(distance);
	return distance;
}

FeatureMatrix *find_centroids(FeatureMatrix *featureMatrix, float **centroids, int *labels, int k, int length)
{
	if (featureMatrix->nFeaturesVectors < 1) {
		return NULL;
	}
	int *vectors_in_cluster = (int *)calloc(k, sizeof *vectors_in_cluster);

	for (int i = 0; i < featureMatrix->nFeaturesVectors; i++) {
		int cluster = labels[i];
		vectors_in_cluster[cluster]++;
		for (int j = 0; j < length; j++) {
			centroids[cluster][j] = featureMatrix->featureVector[i]->features[j];
		}
	}

	for (int i = 0; i < k; i++) {
		for (int j = 0; j < length; j++)
			centroids[i][j] /= vectors_in_cluster[i];
	}

	FeatureMatrix *new_dict = createFeatureMatrix(k);
	for (int i = 0; i < k; i++) {
		new_dict->featureVector[i] = createFeatureVector(centroids[i], length);
	}

	free(vectors_in_cluster);
	return new_dict;
}

TrainingKnowledge* kMeansClustering(FeatureMatrix* featureMatrix, int numberOfCluster)
{
	FeatureMatrix *dict = createFeatureMatrix(numberOfCluster);
	FeatureMatrix *new_dict = NULL;
	int k = 0;
	float dict_distance = 0.0;
	bool *isUsed = (bool *)calloc(featureMatrix->nFeaturesVectors, sizeof *isUsed);
	int *labels = (int *)calloc(featureMatrix->nFeaturesVectors, sizeof *labels);
	// inicializamos o dicionario original para vetores escolhidos aleatoriamente do conjunto
	while (k < numberOfCluster) {
		int randomIndex = RandomInteger(0, featureMatrix->nFeaturesVectors-1);
		if (!isUsed[randomIndex]) {
			dict->featureVector[k] = copyFeatureVector(featureMatrix->featureVector[k]);
			isUsed[randomIndex] = true;
			k++;
		}
	}
	free(isUsed);
	if (featureMatrix->nFeaturesVectors < 1) {
		return NULL;
	}
	float **centroids = (float **)calloc(numberOfCluster, sizeof *centroids); // esse vetor sera reutilizado varias vezes para encontrar centroides, sendo sobrescrito toda vez
	int length = featureMatrix->featureVector[0]->size;
	for (int i = 0; i < k; i++) {
		centroids[i] = (float *)calloc(length, sizeof **centroids);
	}
	do {
		for (int i = 0; i < featureMatrix->nFeaturesVectors; i++) {
			// acha o cluster mais proximo
			// distancia de vetores de 64 dimensoes (histogramas de 4 bins em cada canal, 4 * 4 * 4)
			int closest_cluster = 0;
			float distance_to_closest_cluster = euclidean_distance(featureMatrix->featureVector[i], dict->featureVector[closest_cluster]);
			for (int j = 1; j < numberOfCluster; j++) {
				// calculamos a distancia do vetor i para o centroide j do dicionario
				float distance = euclidean_distance(featureMatrix->featureVector[i], dict->featureVector[j]);
                //printf("%f -> -%f => %d\n", distance, distance_to_closest_cluster, distance < distance_to_closest_cluster);
				if (distance < distance_to_closest_cluster) {
					// se a distancia a esse centroide for menor do que a um anterior, atualizamos o centroide mais proximo
					closest_cluster = j;
					distance_to_closest_cluster = distance;
				}
			}
			// depois de achado
			// labels[i] contem o centroide mais proximo do vetor
			labels[i] = closest_cluster;
		}
		// cria dicionario novo
		// acha novos centroides (soma os vetores e divide pelo numero de vetores somados)
		// criamos um novo conjunto de centroides considerando todos os vetores que foram direcionados para aquele cluster
		new_dict = find_centroids(featureMatrix, centroids, labels, numberOfCluster, length);
		dict_distance = 0.0;
		for (int i = 0; i < dict->nFeaturesVectors; i++) {
			dict_distance += euclidean_distance(dict->featureVector[i], new_dict->featureVector[i]);
		}
		dict_distance /= numberOfCluster; // consideramos a distancia entre dicionarios como a media da distancia entre cada centroide
		destroyFeatureMatrix(&dict); // podemos destruir o dicionario antigo
		dict = new_dict; // agora o novo dicionario eh o dicionario antigo. poetico.
	// repete até a diferença entre o dict velho e o novo ficar menor que um epsilon
	} while (dict_distance > DICT_DIFFERENCE_EPSILON); // ajustar esse epsilon no define acima ate que o programa pare

    TrainingKnowledge* tr = (TrainingKnowledge *)calloc(1, sizeof *tr);
    tr->nlabels = featureMatrix->nFeaturesVectors;
    tr->labels = labels;
    tr->dictionary = dict;

	for (int i = 0; i < numberOfCluster; i++)
		free(centroids[i]);
	free(centroids);
	return tr;
}
