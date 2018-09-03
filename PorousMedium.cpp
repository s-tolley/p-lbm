#include "olb2D.h"
//#include "olb2D.hh"   // use only generic version!
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include "PorousMedium.h"

using namespace olb;
using namespace olb::descriptors;
using namespace olb::graphics;
using namespace std;

typedef double T;
#define DESCRIPTOR ForcedD2Q9Descriptor

/// Función delta de Dirac, distancia medida en celdas
T PorousMedium::deltaDirac(T dist){
  //Valor Absoluto de la distancia
  if (dist < 0) {
    dist = dist * -1;
  }

  if (dist < 1){    // Disatncia entre 0 y 1
    return ((3. - 2.*dist + sqrt(1. + 4.*dist - 4*dist*dist))/8.);
  }

  if (dist < 2){    // Distancia entre 1 y 2
    return ((5. - 2.*dist - sqrt(-7. + 12.*dist - 4.*dist*dist))/8.);
  }

  return 0.;        // Diatancia mayor a 2
}

/// Cálculo de velocidad del entorno a cada punto
void PorousMedium::computeVelocity(SuperLattice2D<T,DESCRIPTOR>& sLattice,
                     LBconverter<T> const & converter){
  /// Recorre un entorno a cada punto y suma las velocidades
  /// de las celdas ponderadas por la distancia
  T velocity[2];
  T v[2];
  T inc = converter.getLatticeL();  // Tamaño de celda
  Cell<T,DESCRIPTOR> cell;
  T x;
  T y;
  T maxX;
  T maxY;
  T diracX;
  T diracY;
  T proximity = influenceRadius * inc;  // Radio de influencia en unidades del Lattice

  ForcedBGKdynamics<T, DESCRIPTOR> d (
    converter.getOmega(),
    instances::getBulkMomenta<T,DESCRIPTOR>()
  );

  for(int i = 0; i < numPoints; i++){
    // Límites del entorno
    x = pointX[i] - proximity;
    y = pointY[i] - proximity;
    maxX = pointX[i] + proximity;
    maxY = pointY[i] + proximity;

    velocity[0] = 0;
    velocity[1] = 0;
    while(x < maxX){
      diracX = deltaDirac((x-pointX[i])/inc);  // Cálculo parcial de función Dirac (distancia en X)
      while(y < maxY){
        if (sLattice.get(x,y,cell)) { // Evaluación de limites del dominio
          d.computeU(cell,v);
          diracY = deltaDirac((y-pointY[i])/inc);  // Cálculo parcial de función Dirac (distancia en Y)
          velocity[0] = velocity[0] + diracX*diracY*v[0];
          velocity[1] = velocity[1] + diracX*diracY*v[1];
        }
        y = y + inc;
      }
      x = x + inc;
      y = pointY[i] - proximity;
    }
    pointVelX[i] = velocity[0];
    pointVelY[i] = velocity[1];
  }
}

/// Cálculo parcial de fuerza, sin ponderar por la distancia
Vector<T,2> PorousMedium::computeForce(LBconverter<T> const & converter,int point){
  T d = converter.latticePorosity(physPermeability);  // Permeabilidad en unidades del Lattice
  T tmp = (-1)*(converter.getLatticeNu()/d);
  Vector<T,2> force (tmp*pointVelX[point],tmp*pointVelY[point]);
  return (force);
}

/// Aplicación de fuerza de cada punto del medio poroso
void PorousMedium::applyForce(SuperLattice2D<T,DESCRIPTOR>& sLattice,
                LBconverter<T> const & converter){
  /// Recorre el entorno de cada punto, calculando y aplicando
  /// la componente de fuerza del medio poroso.
  computeVelocity(sLattice, converter);
  T inc = converter.getLatticeL();  // Tamaño de celda
  Cell<T,DESCRIPTOR> cell;
  T x;
  T y;
  T maxX;
  T maxY;
  T diracX;
  T diracY;
  Vector<T,2> force;
  Vector<T,2> pointForce;
  Vector<T,2> forceX;
  T proximity = influenceRadius * inc;  // Radio de influencia en unidades del Lattice

  for (int i = 0; i < numPoints; i++){
    // Límites del entorno
    x = pointX[i] - proximity;
    y = pointY[i] - proximity;
    maxX = pointX[i] + proximity;
    maxY = pointY[i] + proximity;

    /// Cálculo de fuerza
    force = computeForce(converter, i);

    while(x < maxX){
      diracX = deltaDirac((x-pointX[i])/inc); // Cálculo parcial de función Dirac (distancia en X)
      forceX[0] = force[0]*diracX;  // Aplicación parcial de función Dirac
      forceX[1] = force[1]*diracX;  // Aplicación parcial de función Dirac
      while(y < maxY){
        diracY = deltaDirac((y-pointY[i])/inc); // Cálculo parcial de función Dirac (distancia en Y)
        if (sLattice.get(x,y,cell)){ // Evaluación de limites del dominio
          /// Seteo de fuerza
          T* externalData = cell.getExternal(0);
          externalData[0] += (forceX[0]*diracY);
          externalData[1] += (forceX[1]*diracY);
          sLattice.set(x,y,cell);
        }
        y = y + inc;
      }
      x = x + inc;
      y = pointY[i] - proximity;
    }
  }
}

/// Inicializacón de puntos del medio poroso
void PorousMedium::setPoints(T * pX, T * pY, int number){
  this.pointX = pX;
  this.pointY = pY;
  this.numPoints = num;
  pointVelX = new T[numPoints];
  pointVelY = new T[numPoints];
}

/// Inicializacón de permeabilidad del medio poroso
void PorousMedium::setPermeability(T physPerm){
  this.physPermeability = physPerm;
}

/// Inicializacón de radio de influencia de los puntos del medio poroso
void PorousMedium::setInfluenceRadius(T rad){
  this.influenceRadius = rad;
}
