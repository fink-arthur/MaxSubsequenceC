#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


/***********************
*     Structures       *
***********************/

struct tablo {
  int * tab;
  int size;
};

struct tablo2 {
  int size;
  int * tab;
  int * max;
  int * min;
};

/***********************
*     Print            *
***********************/

void printArray(struct tablo * tmp) {
  printf("---- Array of size %i ---- \n", tmp->size);
  int size = tmp->size;
  int i;
  for (i = 0; i < size; ++i) {
    printf("%i ", tmp->tab[i]);
  }
  printf("\n");
}

void printArray2(struct tablo2 * tmp) {
  printf("---- Array of size %i ---- \n", tmp->size);
  int size = tmp->size;
  int i;
  for (i = 0; i < size; ++i) {
    printf("{%i %i %i} ", tmp->tab[i], tmp->min[i], tmp->max[i]);
  }
  printf("\n");
}


/***********************
*     Allocation       *
***********************/

struct tablo2 * allocateTablo2(int size) {
  // Allocation d'une struct tablo2 avec mise a 0 de tous les elements
  struct tablo2 * tmp = malloc(sizeof(struct tablo2));
  tmp->size = size;
  tmp->tab = malloc(size*sizeof(int));
  tmp->max = malloc(size*sizeof(int));
  tmp->min = malloc(size*sizeof(int));
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    tmp->tab[i] = 0;
    tmp->max[i] = 0;
    tmp->min[i] = 0;
  }
  return tmp;
}

struct tablo * allocateTablo(int size) {
  // Allocation d'une struct tablo avec une mise a 0 de tous les elements
  struct tablo * tmp = malloc(sizeof(struct tablo));
  tmp->size = size;
  tmp->tab = malloc(size*sizeof(int));
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    tmp->tab[i] = 0;
  }
  return tmp;
}

/***********************
*  Max Pre/Suf-fix     *
***********************/

void montee_max(struct tablo * sum_prefix_final, struct tablo * sum_suffix_final, struct tablo * max_prefix, struct tablo * max_suffix) {
  // La montee pour le prefixe et le suffixe max en meme temps
  #pragma omp parallel for
  for (int i = 0; i < sum_prefix_final->size / 2; i++) {
    int indice = sum_prefix_final->size / 2 + i;
    int indice_suffix = (sum_prefix_final->size - 1) - i;
    max_suffix->tab[max_suffix->size / 2 + i] = sum_prefix_final->tab[indice_suffix];
    max_prefix->tab[max_prefix->size / 2 + i] = sum_suffix_final->tab[indice];
  }

  for (int l = log2(sum_prefix_final->size) - 1; l > 0; l--) {
    #pragma omp parallel for
    for (int j = 1 << (l-1) ; j <= (1 << l) - 1; j++) {
      max_prefix->tab[j] = fmax(max_prefix->tab[2*j],max_prefix->tab[2*j+1]);
      max_suffix->tab[j] = fmax(max_suffix->tab[2*j],max_suffix->tab[2*j+1]);
    }
  }
}

void descente_max(struct tablo * max_prefix, struct tablo * max_prefix_final, struct tablo * max_suffix, struct tablo * max_suffix_final) {
  // La descente pour le prefixe et le suffixe max en meme temps
  max_prefix_final->tab[1] = INT_MIN;
  max_suffix_final->tab[1] = INT_MIN;
  for (int l = 2; l <= log2(max_prefix->size); l++) {
    #pragma omp parallel for
    for (int j = 1 << (l-1); j <= (1 << l) - 1; j++) {
      if (j % 2 == 0) {
        max_prefix_final->tab[j] = max_prefix_final->tab[j/2];
        max_suffix_final->tab[j] = max_suffix_final->tab[j/2];
      }
      else {
        max_prefix_final->tab[j] = fmax(max_prefix_final->tab[j/2],max_prefix->tab[j-1]);
        max_suffix_final->tab[j] = fmax(max_suffix_final->tab[j/2],max_suffix->tab[j-1]);
      }
    }
  }
}

void final_max(struct tablo * max_prefix, struct tablo * max_prefix_final, struct tablo * max_suffix, struct tablo * max_suffix_final) {
  // La partie finale pour le prefixe et le suffixe max en meme temps
  int size = max_prefix->size / 2;
  #pragma omp parallel for
  for (int j = size; j <= 2 * size - 1; j++) {
    max_prefix_final->tab[j] = fmax(max_prefix_final->tab[j],max_prefix->tab[j]);
    max_suffix_final->tab[j] = fmax(max_suffix_final->tab[j],max_suffix->tab[j]);
  }
}

/***********************
*      Sum Prefix      *
***********************/

void montee_sum(struct tablo * source, struct tablo * sum_prefix) {
  // La montee pour le prefixe somme
  #pragma omp parallel for
  for (int i = 0; i < source->size; i++) {
    sum_prefix->tab[sum_prefix->size / 2 + i] = source->tab[i];
  }

  for (int l = log2(source->size); l > 0; l--) {
    #pragma omp parallel for
    for (int j = 1 << (l-1) ; j <= (1 << l) - 1; j++) {
      sum_prefix->tab[j] = sum_prefix->tab[2*j] + sum_prefix->tab[2*j+1];
    }
  }
}

void descente_sum(struct tablo * sum_prefix, struct tablo * sum_prefix_final) {
  // La descente pour le prefixe somme
  sum_prefix_final->tab[1] = 0;
  for (int l = 2; l <= log2(sum_prefix->size); l++) {
    #pragma omp parallel for
    for (int j = 1 << (l-1); j <= (1 << l) - 1; j++) {
      if (j % 2 == 0) {
        sum_prefix_final->tab[j] = sum_prefix_final->tab[j/2];
      }
      else {
        sum_prefix_final->tab[j] = sum_prefix_final->tab[j/2] + sum_prefix->tab[j-1];
      }
    }
  }
}

void final_sum(struct tablo * sum_prefix, struct tablo * sum_prefix_final) {
  // La partie finale pour le prefixe somme
  int size = sum_prefix->size / 2;
  #pragma omp parallel for
  for (int j = size; j <= 2 * size - 1; j++) {
    sum_prefix_final->tab[j] = sum_prefix_final->tab[j] + sum_prefix->tab[j];
  }
}

/***********************
*  Sum suffix          *
***********************/

void calcul_sum_suffix(struct tablo * sum_prefix_final, struct tablo * sum_suffix_final, struct tablo * source) {
  // Le calcul en une passe du suffixe somme en se basant sur le prefixe somme
  int maximum_prefix = sum_prefix_final->tab[sum_prefix_final->size - 1];
  sum_suffix_final->tab[(sum_suffix_final->size/2)] = maximum_prefix;
  #pragma omp parallel for
  for (int i = (sum_suffix_final->size/2) + 1; i < sum_suffix_final->size; i++) {
      sum_suffix_final->tab[i] = maximum_prefix - sum_prefix_final->tab[i-1];
  }
}

/***********************
*     Calcul du ssm    *
***********************/

void ssm(struct tablo * max_prefix_final, struct tablo * sum_prefix_final, struct tablo * max_suffix_final, struct tablo * sum_suffix_final, struct tablo * source, struct tablo * res) {
  // Le calcul final de la sous sequence maximale
  int size = source->size;
  for (int i = 0; i < size; i++) {
    int indice = i + source->size;
    int indice_suffix = (size - 1) - i + source->size;
    res->tab[i] = max_prefix_final->tab[indice] - sum_suffix_final->tab[indice] + max_suffix_final->tab[indice_suffix] - sum_prefix_final->tab[indice] + source->tab[i];
  }
}

/***********************
*  Calcul du max       *
***********************/

void montee_res(struct tablo * source, struct tablo2 * res) {
  // La montee pour trouver le max ainsi que les bornes de celui-ci dans la ssm
  #pragma omp parallel for
  for (int i = 0; i < source->size; i++) {
    res->tab[res->size / 2 + i] = source->tab[i];
    res->min[res->size / 2 + i] = i;
    res->max[res->size / 2 + i] = i;
  }

  for (int l = log2(source->size); l > 0; l--) {
    #pragma omp parallel for
    for (int j = 1 << (l-1) ; j <= (1 << l) - 1; j++) {
      res->tab[j] = fmax(res->tab[2*j], res->tab[2*j+1]);
      // Si on a le meme max, on regarde en details les indices
      if (res->tab[2*j] == res->tab[2*j+1]) {
        res->min[j] = fmin(res->min[2*j], res->min[2*j+1]);
        // On regarde que les chaines se suivent bien dans le tableau
        if (res->min[2*j+1] - res->max[2*j] == 1){
          res->max[j] = fmax(res->max[2*j], res->max[2*j+1]);
        }
        // Sinon on prend la premiere en commencant de la droite
        else {
          res->max[j] = fmin(res->max[2*j], res->max[2*j+1]);

        }
      }
      // sinon on prends celui du max
      else {
        if (res->tab[j] == res->tab[2*j+1]) {
          res->min[j] = res->min[2*j+1];
          res->max[j] = res->max[2*j+1];
        }
        else {
          res->min[j] = res->min[2*j];
          res->max[j] = res->max[2*j];
        }
      }
    }
  }
}

/***********************
*    Lecture fichier   *
***********************/

void lecture(struct tablo * source, char * nom) {
  // Lecture du fichier avec un getline qui est lent mais extremement facile d'utilisation
  FILE * fp = fopen(nom, "r");

  char * line = NULL;
  size_t len = 0;

  int fd = fileno(fp);
  struct stat buf;
  fstat(fd, &buf);
  int size = buf.st_size;

  // la taille se base sur le fait que seulement si les nombres ont 1 chiffres nous devrons reallouer
  // sinon le tableau sera plus grand que necessaire de plus on sait que le tableau a un taille 2^n
  int realloc_size = pow(2,ceil(log2(size / 2)));

  source->tab = malloc(realloc_size * sizeof(int));

  getline(&line, &len, fp);

  fclose(fp);

  int i = 0;
  char * token;
  char *line2 = line; // pour eviter la fuite memoire

  // on split la ligne avec le caractere espace
  while ((token = strsep(&line2, " ")) != NULL) {
    source->tab[i] = atoi(token);
    i++;
    // on réalloue si le tableau depasse la taille et on maj la taille
    if (i == realloc_size) {
      realloc_size = 2 * realloc_size;
      source->tab = realloc(source->tab, realloc_size * sizeof(int));
    }
  }
  // On met la taille du tableau en dur dans la structure
  // les cases en trop sont ignoré
  source->size = i;

  free(line);
  free(token);
}

int main(int argc, char **argv) {

  /***********************
  *  Ouverture fichier   *
  ***********************/

  struct tablo * source = malloc(sizeof(struct tablo));

  lecture(source, argv[1]);

  /***********************
  *  Tab allocations     *
  ***********************/

  struct tablo * sum_prefix = allocateTablo(source->size*2);
  struct tablo * sum_suffix2 = allocateTablo(source->size*2);
  struct tablo * sum_prefix_final = allocateTablo(source->size*2);

  struct tablo * max_prefix = allocateTablo(source->size*2);
  struct tablo * max_suffix = allocateTablo(source->size*2);
  struct tablo * max_prefix_final = allocateTablo(source->size*2);
  struct tablo * max_suffix_final = allocateTablo(source->size*2);

  struct tablo * res = allocateTablo(source->size);
  struct tablo2 * maximum = allocateTablo2(res->size * 2);

  /***********************
  *  Pre/Suf-fix calcul  *
  ***********************/

  montee_sum(source, sum_prefix);
  descente_sum(sum_prefix, sum_prefix_final);
  final_sum(sum_prefix, sum_prefix_final);

  calcul_sum_suffix(sum_prefix_final, sum_suffix2, source);

  montee_max(sum_prefix_final, sum_suffix2, max_prefix, max_suffix);
  descente_max(max_prefix, max_prefix_final, max_suffix, max_suffix_final);
  final_max(max_prefix,max_prefix_final, max_suffix, max_suffix_final);

  /***********************
  *  ssm et max calcul   *
  ***********************/

  ssm(max_prefix_final, sum_prefix_final, max_suffix_final, sum_suffix2, source, res);

  montee_res(res,maximum);

  /***********************
  *  affichage debug     *
  ***********************/

  /*printf("RES\n");
  printArray(res);
  printf("PSUM\n");
  printArray(sum_prefix);
  printArray(sum_prefix_final);
  printf("SSUM\n");
  printArray(sum_suffix2);
  printf("PMAX\n");
  printArray(max_prefix);
  printArray(max_prefix_final);
  printf("SMAX\n");
  printArray(max_suffix);
  printArray(max_suffix_final);
  printf("INDICE\n");
  printArray2(maximum);
  printf("resultat\n");*/

  /***********************
  *    Affichage res     *
  ***********************/

  printf("%d", maximum->tab[1]);
  for (int i = maximum->min[1]; i <= maximum->max[1]; i++){
    printf(" %d", source->tab[i]);
  }
  printf("\n");

  /***********************
  * Garbage collection   *
  ***********************/

  free(source->tab);
  free(source);
  free(sum_prefix->tab);
  free(sum_prefix);
  free(sum_suffix2->tab);
  free(sum_suffix2);
  free(sum_prefix_final->tab);
  free(sum_prefix_final);
  free(max_prefix->tab);
  free(max_prefix);
  free(max_suffix->tab);
  free(max_suffix);
  free(max_prefix_final->tab);
  free(max_prefix_final);
  free(max_suffix_final->tab);
  free(max_suffix_final);
  free(res->tab);
  free(res);
  free(maximum->tab);
  free(maximum->max);
  free(maximum->min);
  free(maximum);
}
