#include "olb2D.h"
#include "olb2D.hh"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "io/xmlReader.h"
#include <omp.h>
#include "vtkCommonCoreModule.h"
#include "vtkSmartPointer.h"
#include "vtkCellArray.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyDataWriter.h"
#include "vtkNew.h"
#include "PorousMedium.h"

using namespace olb;
using namespace olb::descriptors;
using namespace olb::graphics;
using namespace std;

typedef double T;
#define DESCRIPTOR ForcedD2Q9Descriptor

/// Parámetros de la simulación
T lx;         // largo del canal
T ly;         // alto del canal
int N;        // resolución del modelo
int M;        // refinamiento del tiempo de disctretización (time discretization refinement)
T maxPhysT;   // tiempo de simulación en segundos
T Nu;         // viscosidad
T lattU;      // velocidad del Lattice
T delta;      // tamaño de celda

/// Checkpoints
int useCheckpoint = 0;
int checkpointIteration = 0;
string checkpointFile = "";

/// Etapas de velocidad (empinada y leve)
int phase1;             // duración de fase de pendiente empinada
int phase2;             // duración de fase de pendiente leve
T velocityPhase1 = 0.;  // velocidad al finalizar la etapa 1
T maxVelocity = 0.;     // velocidad al finalizar la etapa 2
T velocity = 0.;

/// Extracción de resultados
T extPointPosX = 0;
T extPointPosY = 0;
T extSlicePosX = 0;
T extReyX = 0;
int extNumIterations = 10000;

/// Guardado en VTK
vtkPolyData* polyCell;
vtkDoubleArray* polyCellVelX;
vtkDoubleArray* polyCellVelY;
vtkDoubleArray* polyCellRho;

vtkPolyData* slices;
vtkDoubleArray* polySliceVelX;
vtkDoubleArray* polySliceVelY;
vtkDoubleArray* polySliceRho;

/// Inicialización de la Geometría
void prepareGeometry(LBconverter<T> const& converter,
                     SuperGeometry2D<T>& superGeometry) {

  OstreamManager clout(std::cout,"prepareGeometry");
  clout << "Prepare Geometry ..." << std::endl;

  Vector<T,2> extend(lx,ly);
  Vector<T,2> origin;

  superGeometry.rename(0,2);                      // renombra todo el dominio a 2

  superGeometry.rename(2,1,1,1);                  // renombra todo de 2 a 1 exceptuando los bordes

  /// Definición del número de material de la entrada
  extend[0] = 2.*converter.getLatticeL();
  origin[0] = -converter.getLatticeL();
  IndicatorCuboid2D<T> inflow(extend, origin);
  superGeometry.rename(2,3,1,inflow);             // renombra de 2 a 3 las celdas dentro de inflow

  /// Definición del número de material de la salida
  origin[0] = lx-converter.getLatticeL();
  IndicatorCuboid2D<T> outflow(extend, origin);
  superGeometry.rename(2,4,1,outflow);            // renombra de 2 a 4 las celdas dentro de outflow

  /// Eliminación de voxels innecesarios fuera de la superficie
  superGeometry.clean();
  superGeometry.checkForErrors();

  superGeometry.print();

  clout << "Prepare Geometry ... OK" << std::endl;
}

/// Inicialización del Lattice
void prepareLattice(LBconverter<T> const& converter,
                    SuperLattice2D<T, DESCRIPTOR>& sLattice,
                    Dynamics<T, DESCRIPTOR>& bulkDynamics,
                    sOnLatticeBoundaryCondition2D<T,DESCRIPTOR>& sBoundaryCondition,
                    SuperGeometry2D<T>& superGeometry ) {

  OstreamManager clout(std::cout,"prepareLattice");
  clout << "Prepare Lattice ..." << std::endl;

  T omega = converter.getOmega();

  /// Material=0 --> no hace nada
  sLattice.defineDynamics(superGeometry, 0, &instances::getNoDynamics<T, DESCRIPTOR>());

  /// Material=1 --> dinámica de fluido (canal)
  sLattice.defineDynamics(superGeometry, 1, &bulkDynamics);

  /// Material=2 --> dinámica de bounce back (bordes)
  sLattice.defineDynamics(superGeometry, 2, &instances::getBounceBack<T, DESCRIPTOR>());

  /// Material=3 --> dinámica de fluido (entrada del canal)
  sLattice.defineDynamics(superGeometry, 3, &bulkDynamics);   /// agregado

  /// Material=4 --> dinámica de fluido (salida del canal)
  sLattice.defineDynamics(superGeometry, 4, &bulkDynamics);   /// agregado

  /// Definición de condiciones de contorno
  sBoundaryCondition.addVelocityBoundary(superGeometry, 3, omega);
  sBoundaryCondition.addPressureBoundary(superGeometry, 4, omega);

  /// Condiciones iniciales
  AnalyticalConst2D<T,T> rhoF(1);
  std::vector<T> velocity(2,T(0));
  AnalyticalConst2D<T,T> uF(velocity);

  /// Inicialización de valores de velocidad y densidad
  sLattice.defineRhoU(superGeometry, 1, rhoF, uF);
  sLattice.iniEquilibrium(superGeometry, 1, rhoF, uF);
  sLattice.defineRhoU(superGeometry, 3, rhoF, uF);
  sLattice.iniEquilibrium(superGeometry, 3, rhoF, uF);
  sLattice.defineRhoU(superGeometry, 4, rhoF, uF);
  sLattice.iniEquilibrium(superGeometry, 4, rhoF, uF);

  sLattice.initialize();

  clout << "Prepare Lattice ... OK" << std::endl;
}

/// Eliminación de archivos de otras ejecuciones
void deleteOldFiles(){
  remove("VelocidadPromedio.txt");
  remove("Reynolds.txt");
  remove("DesvioEstandar.txt");
  remove("EvolucionPunto.vtk");
  remove("EvolucionPerfil.vtk");
  char const * aux = checkpointFile.c_str();
  remove(aux);
}

/// Guardado de Checkpoint
void saveState(SuperLattice2D<T,DESCRIPTOR>& sLattice, LBconverter<T>& converter){
  OstreamManager clout(std::cout,"Save");
  clout << "Saving system state" << std::endl;
  sLattice.save("porous.checkpoint");

  /// Guardado de valores de velocidad e iteración finales
  Cell<T,DESCRIPTOR> cell;
  T vel[2];
  if(sLattice.get(0,ly/2,cell)){
    ForcedBGKdynamics<T, DESCRIPTOR> d (
      converter.getOmega(),
      instances::getBulkMomenta<T,DESCRIPTOR>()
    );
    d.computeU(cell,vel);
    FILE *output = fopen("checkpointData.txt","w");
    fprintf(output, "Iteracion: %d, Velocidad X: %lf\n", converter.numTimeSteps(maxPhysT)-1, vel[0]);
    fclose(output);
  }
}

/// Cargado de valores desde Checkpoint
void loadCheckpoint(SuperLattice2D<T,DESCRIPTOR>& sLattice, int& iT){
  OstreamManager clout(std::cout,"checkpoint");
  clout << "Loading Checkpoint at t=" << checkpointIteration << std::endl;
  sLattice.load(checkpointFile);
  iT = checkpointIteration;
}

/// Guardado de valores del canal
void getResults(SuperLattice2D<T,DESCRIPTOR>& sLattice,
                LBconverter<T>& converter, int iT,
                SuperGeometry2D<T>& superGeometry, Timer<T>& timer) {

  SuperVTKwriter2D<T> vtkWriter("plbm");
  SuperLatticePhysVelocity2D<T, DESCRIPTOR> velocity(sLattice, converter);
  SuperLatticePhysPressure2D<T, DESCRIPTOR> pressure(sLattice, converter);
  vtkWriter.addFunctor( velocity );
  vtkWriter.addFunctor( pressure );

  const int vtkIter  = 20;                          // frecuencia de gardado
  const int statIter = converter.numTimeSteps(.1);  // frecuencia de estadísticas
  const int startOnIter = 0;                        // iteración de comienzo de guardado

  if (iT==0) {
    /// Creación de archivo inicial para converter
    writeLogFile(converter, "plbm");

    /// Creación de archivos iniciales para geometría
    SuperLatticeGeometry2D<T, DESCRIPTOR> geometry(sLattice, superGeometry);
    SuperLatticeCuboid2D<T, DESCRIPTOR> cuboid(sLattice);
    SuperLatticeRank2D<T, DESCRIPTOR> rank(sLattice);
    superGeometry.rename(0,2);
    vtkWriter.write(geometry);
    vtkWriter.write(cuboid);
    vtkWriter.write(rank);

    vtkWriter.createMasterFile();
  }

  /// Escritura de archivos vtk y "profile text file" TODO
  if ((iT >= startOnIter)&&(iT%vtkIter==0)) {
    vtkWriter.write(iT);

    ofstream *ofile = 0;
    if (singleton::mpi().isMainProcessor()) {
      ofile = new ofstream((singleton::directories().getLogOutDir()+"centerVel.dat").c_str());
    }
    T Ly = converter.numCells(ly);
    for (int iY=0; iY<=Ly; ++iY) {
      T dx = converter.getDeltaX();
      const T maxVelocity = converter.physVelocity(converter.getLatticeU());
      T point[2]={T(),T()};
      point[0] = lx/2.;
      point[1] = (T)iY/Ly;
      const T radius = ly/2.;
      std::vector<T> axisPoint(2,T());
      axisPoint[0] = lx/2.;
      axisPoint[1] = ly/2.;
      std::vector<T> axisDirection(2,T());
      axisDirection[0] = 1;
      axisDirection[1] = 0;
      Poiseuille2D<T> uSol(axisPoint, axisDirection, maxVelocity, radius);
      T analytical[2] = {T(),T()};
      uSol(analytical,point);
      SuperLatticePhysVelocity2D<T, DESCRIPTOR> velocity(sLattice, converter);
      AnalyticalFfromSuperLatticeF2D<T, DESCRIPTOR> intpolateVelocity(velocity, true);
      T numerical[2] = {T(),T()};
      intpolateVelocity(numerical,point);
      if (singleton::mpi().isMainProcessor()) {
        *ofile << iY*dx << " " << analytical[0]
               << " " << numerical[0] << "\n";
      }
    }
    delete ofile;
  }

  /// Escritura por consola
  if (iT%statIter==0) {
    /// Valores del Timer
    timer.update(iT);
    timer.printStep();

    /// Estadisticas del Lattice
    sLattice.getStatistics().print(iT,converter.physTime(iT));
  }
}

/// Aumento de velocidad
void scaleVelocity(SuperLattice2D<T, DESCRIPTOR>& sLattice,
                   LBconverter<T> const& converter, int iT,
                   SuperGeometry2D<T>& superGeometry){

  T distance2Wall = converter.getLatticeL()/2.;
  int iTupdate = 5;       // frecuencia de actualización en iteraciones
  if(iT%iTupdate==0){
    if(iT < phase1){      // aumento durante etapa 1
      velocity = velocity + (velocityPhase1-velocity)/(phase1-iT) * iTupdate;
    }
    else{
      if(iT < phase2){    // aumento durante etapa 2
        velocity = velocity + (maxVelocity-velocity)/(phase2-iT) * iTupdate;
      }
    }
  Poiseuille2D<T> poiseuilleU(superGeometry, 3, velocity, distance2Wall);
  sLattice.defineU(superGeometry, 3, poiseuilleU);
  }
}

/// Seteo de fuerzas externas en cero
void resetForces(SuperLattice2D<T,DESCRIPTOR>& sLattice,
                 LBconverter<T> const & converter,
                 SuperGeometry2D<T>& superGeometry){

  std::vector<T> forceA(2,T());
  forceA[0] = 0.;
  forceA[1] = 0.;
  AnalyticalConst2D<T,T> forceTerm(forceA);     // término de fuerza en cero

  std::vector<T> extend(2,T());
  extend[0] = lx;
  extend[1] = ly;
  std::vector<T> origin(2,T());
  IndicatorCuboid2D<T> cuboid(extend, origin);  // indicador que abarca todo el canal
  sLattice.defineExternalField(superGeometry, cuboid, DESCRIPTOR<T>::ExternalField::forceBeginsAt, DESCRIPTOR<T>::ExternalField::sizeOfForce, forceTerm); // seteo del término en el área abarcada por el indicador
}

/// Carga de Parámetros de geometría desde xml
  /// Dimensiones del canal, los puntos del medio poroso y la permeabilidad
void setGeometryParameters(PorousMedium & pm){
  string fName("geometria.xml");
  XMLreader param (fName);
  lx = param["parametros"]["lx"].get<T>();
  ly = param["parametros"]["ly"].get<T>();

  int numPoints = param["parametros"]["numPoints"].get<int>();
  T physPermeability = param["parametros"]["physPermeability"].get<T>();

  /// Puntos
  T * pointX = new T[numPoints];
  T * pointY = new T[numPoints];

  char * p = new char[100]; // armado de string <px> </px>
  p[0]= 'p';
  char * c = new char[99];
  for(int i = 0; i < numPoints; i++){
    sprintf(c, "%d", i);
    for (int j = 0; j < 99; j++){
      p[j+1] = c[j];
    }
    pointX[i] = param["puntos"][p]["x"].get<T>();
    pointY[i] = param["puntos"][p]["y"].get<T>();
  }
  pm.setPoints(pointX,pointY,numPoints);
  pm.setPermeability(physPermeability);
  delete [] p;
  p = NULL;
  delete [] c;
  c = NULL;
  delete [] pointX;
  pointX = NULL;
  delete [] pointY;
  pointY = NULL;
}

/// Carga de Parámetros de disctretización desde xml
/// N, M, velocidad del lattice, delta y Radio de influencia
void setDiscParameters(PorousMedium & pm){
  string fName("discretizacion.xml");
  XMLreader param (fName);

  N = param["N"].get<int>();
  M = param["M"].get<int>();
  T influenceRadius = param["influenceRadius"].get<T>();
  delta = param["delta"].get<T>();
  lattU = param["latticeU"].get<T>();
  pm.setInfluenceRadius(influenceRadius);
}

/// Carga de Parámetros de simulación desde xml
/// tiempo de simulación y viscosidad del fluido
void setSimulationParameters(){
  string fName("simulacion.xml");
  XMLreader param (fName);

  maxPhysT = param["maxPhysT"].get<T>();
  Nu = param["Nu"].get<T>();
}

/// Carga de Parámetros de setup desde xml
/// Evolución de la velocidad de entrada
void setSetupParameters(LBconverter<T> const & converter){
  string fName("setup.xml");
  XMLreader param (fName);

  velocity = param["velIni"].get<T>();
  velocityPhase1 = param["velMaxPahse1"].get<T>();
  maxVelocity = param["velMax"].get<T>();
  T aux1 = param["phase1"].get<T>();      // duración de etapa 1 en segundos
  T aux2 = param["phase2"].get<T>();      // duración de etapa 2 en segundos
  phase1 = converter.numTimeSteps(aux1);  // traducción de segundos a iteraciones
  phase2 = converter.numTimeSteps(aux2);  // traducción de segundos a iteraciones
}

/// Carga de Parámetros de checkpoint desde xml
/// Nombre de archivo, iteración inicial y si se cargará desde el checkpoint
void setCheckpointParameters(){
  string fName("checkpoint.xml");
  XMLreader param (fName);

  useCheckpoint = param["use"].get<int>();
  checkpointFile = param["file"].get<string>();
  checkpointIteration = param["iteration"].get<int>();
}

/// Carga de Parámetros de extracción desde xml
/// Posiciones del punto, el corte y el reynolds, y número de iteraciones a guardar
void setExtractionParameters(){
  string fName("extraccion.xml");
  XMLreader param (fName);

  extPointPosX = param["pointX"].get<T>();
  extPointPosY = param["pointY"].get<T>();
  extSlicePosX = param["sliceX"].get<T>();
  extReyX = param["reyX"].get<T>();
  extNumIterations = param["iterations"].get<int>();
}

/// Cálculo de Reynolds
T computeReynolds(T x, SuperLattice2D<T,DESCRIPTOR>& sLattice, LBconverter<T> const & converter){
  T inc = converter.getLatticeL();        // tamaño de celda
  T y = inc;
  Cell<T,DESCRIPTOR> cell;
  std::vector<T> velocity(2,T());
  T v[2];
  velocity[0] = 0;

  while(y < ly){                          // iteración sobre corte en X
    if (sLattice.get(x,y,cell)){

      ForcedBGKdynamics<T, DESCRIPTOR> d (
        converter.getOmega(),
        instances::getBulkMomenta<T,DESCRIPTOR>()
      );
      d.computeU(cell,v);
      velocity[0] = velocity[0] + v[0];   //suma de velocidades en en eje X
    }
    y = y + inc;
  }

  velocity[0] = velocity[0]/(converter.numCells(ly)-1) * converter.physVelocity(); // promedio de velocidad

  return (velocity[0]*ly/converter.getCharNu());  // cálculo de reynolds
}

/// Extracción de Reynolds
void recordReynolds(int iT, T x, SuperLattice2D<T,DESCRIPTOR>& sLattice, LBconverter<T> const & converter){
  const int start = converter.numTimeSteps(maxPhysT) - extNumIterations;    // iteración de comienzo
  const int reIter  = 2;        // frecuencia de extracción en iteraciones
  if (iT > start){
    if (iT%reIter==0) {
      FILE *output = fopen("Reynolds.txt","a");
      fprintf(output, "Iteracion: %d; Reynolds: %lf\n", iT, computeReynolds(extReyX, sLattice, converter));
      fclose(output);
    }
  }
}

/// Extracción de valores del punto
void updatePointEvolution(int iT, SuperLattice2D<T,DESCRIPTOR>& sLattice, LBconverter<T> const & converter){
  const int start = converter.numTimeSteps(maxPhysT) - extNumIterations;    // iteración de comienzo
  if (iT >= start){
    T v[2];
    T rho;
    Cell<T,DESCRIPTOR> cell;
    ForcedBGKdynamics<T, DESCRIPTOR> d (
      converter.getOmega(),
      instances::getBulkMomenta<T,DESCRIPTOR>()
    );
    if(sLattice.get(extPointPosX,extPointPosY,cell)){
      d.computeRhoU(cell,rho,v);
      polyCellVelX->InsertNextValue(v[0]);      // guardado de valores en estructura auxiliar
      polyCellVelY->InsertNextValue(v[1]);
      polyCellRho->InsertNextValue(rho);
    }
  }
}

/// Inicialización de estructuras de guardado del punto
void initExtractionPoint(){
  polyCell = vtkPolyData::New();
  vtkPoints* polyCellPoints = vtkPoints::New();
  vtkCellArray* polyCellVerts = vtkCellArray::New();
  vtkFloatArray* pcoordsCell = vtkFloatArray::New();

  pcoordsCell->SetNumberOfComponents(3);
  pcoordsCell->SetNumberOfTuples(extNumIterations);
  for(int t = 0; t < extNumIterations; t++){
    float dcell[2]= {t,0};
    pcoordsCell->SetTuple(t,dcell);
  }

  // seteo de puntos
  polyCellPoints->SetData(pcoordsCell);
  pcoordsCell->Delete();

  // seteo de vertices
  for(int i = 0; i < extNumIterations; i++){
    polyCellVerts->InsertNextCell(1);
    polyCellVerts->InsertCellPoint(i);
  }

  // seteo de arreglos
  polyCellVelX = vtkDoubleArray::New();
  polyCellVelX->SetName("VelX");
  polyCellVelY = vtkDoubleArray::New();
  polyCellVelY->SetName("VelY");
  polyCellRho = vtkDoubleArray::New();
  polyCellRho->SetName("Rho");

  // seteo de polyData
  polyCell->SetPoints(polyCellPoints);
  polyCell->SetVerts(polyCellVerts);
  polyCellPoints->Delete();
  polyCellVerts->Delete();
}

/// Extracción de valores del corte
void updateProfileEvolution(int iT, SuperLattice2D<T,DESCRIPTOR>& sLattice, LBconverter<T> const & converter){
  const int start = converter.numTimeSteps(maxPhysT) - extNumIterations;  // iteración de comienzo
  if (iT >= start){
    T v[2];
    T rho;
    Cell<T,DESCRIPTOR> cell;
    ForcedBGKdynamics<T, DESCRIPTOR> d (
      converter.getOmega(),
      instances::getBulkMomenta<T,DESCRIPTOR>()
    );

    T y = 0;
    T  inc = converter.getLatticeL();  // tamaño de celda
    while(y < ly){
      if(sLattice.get(extSlicePosX,y,cell)){
        d.computeRhoU(cell,rho,v);
        polySliceVelX->InsertNextValue(v[0]);   // guardado de valores en estructura auxiliar
        polySliceVelY->InsertNextValue(v[1]);
        polySliceRho->InsertNextValue(rho);
      }
      y = y + inc;
    }
  }
}

/// Inicialización de estructuras de guardado del corte
void initSlices(LBconverter<T> const & converter){
  int cells = converter.numCells(ly);

  int numPoints = extNumIterations*cells;
  slices = vtkPolyData::New();
  vtkPoints* polySlicePoints = vtkPoints::New();
  vtkCellArray* polySliceVerts = vtkCellArray::New();
  vtkFloatArray* pcoords = vtkFloatArray::New();

  // seteo de coordenadas
  pcoords->SetNumberOfComponents(3);
  pcoords->SetNumberOfTuples(numPoints);
  for(int t = 0; t < extNumIterations; t++){
    for(int y = 0; y < cells; y++){
        float f[2] = {t,y};
        pcoords->SetTuple(t*cells +y, f);
      }
  }

  // seteo de puntos
  polySlicePoints->SetData(pcoords);
  pcoords->Delete();

  // seteo de vertices
  for(int i = 0; i < numPoints; i++){
    polySliceVerts->InsertNextCell(1);
    polySliceVerts->InsertCellPoint(i);
  }

  // seteo de arreglos
  polySliceVelX = vtkDoubleArray::New();
  polySliceVelX->SetName("VelX");
  polySliceVelY = vtkDoubleArray::New();
  polySliceVelY->SetName("VelY");
  polySliceRho = vtkDoubleArray::New();
  polySliceRho->SetName("Rho");

  // seteo de polyData
  slices->SetPoints(polySlicePoints);
  slices->SetVerts(polySliceVerts);
  polySlicePoints->Delete();
  polySliceVerts->Delete();
}

/// Guardado de valores extraídos (punto o corte) de estructura auxiliar a archivo vtk
void recordData(vtkPolyData* container, vtkDoubleArray* velX, vtkDoubleArray* velY, vtkDoubleArray* rho, string filename){
  container->GetPointData()->AddArray(velX);
  container->GetPointData()->AddArray(velY);
  container->GetPointData()->AddArray(rho);
  velX->Delete();
  velY->Delete();
  rho->Delete();

  vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
  writer->SetInputData(container);
  writer->SetFileName(filename);
  writer->Write();
  slices->Delete();
}

int main(int argc, char* argv[]) {
  /// ===============================
  /// === Etapa 1: Inicialización ===
  /// ===============================
  olbInit(&argc, &argv);
  singleton::directories().setOutputDir("./tmp/");
  OstreamManager clout(std::cout,"main");

  #ifdef PARALLEL_MODE_MPI
    int ierr = MPI_Comm_rank(MPI_COMM_WORLD, &node);
    if (ierr) printf("> > > Error en MPI < < <\n");
    else printf("nodo: %d\n", node);
  #endif

  /// Carga de parámetros desde xml
  PorousMedium pMedium;
  setGeometryParameters(pMedium);
  setDiscParameters(pMedium);
  setSimulationParameters();
  setCheckpointParameters();
  setExtractionParameters();

  /// Inicialización del Conversor de unidades
  LBconverter<T> converter(
    (int) 2,
    (T)   delta/N,
    (T)   lattU/M,
    (T)   Nu,
    (T)   0.1,
    (T)   0.2
  );
  converter.print();

  /// Carga de parámetros desde xml e inicialización de estructuras auxiliares
  setSetupParameters(converter);
  initSlices(converter);
  initExtractionPoint();

  /// Inicialización de la Geometría
  std::vector<T> extend(2,T());
  extend[0] = lx;
  extend[1] = ly;
  std::vector<T> origin(2,T());
  IndicatorCuboid2D<T> cuboid(extend, origin);
  #ifdef PARALLEL_MODE_MPI
    const int noOfCuboids = singleton::mpi().getSize();
  #else
    const int noOfCuboids = 7;
  #endif
  CuboidGeometry2D<T> cuboidGeometry(cuboid, converter.getLatticeL(), noOfCuboids);
  HeuristicLoadBalancer<T> loadBalancer(cuboidGeometry);
  SuperGeometry2D<T> superGeometry(cuboidGeometry, loadBalancer, 2);
  prepareGeometry(converter, superGeometry);

  /// Inicialización del Lattice
  SuperLattice2D<T, DESCRIPTOR> sLattice(superGeometry);
  /// Instanciación de la dinámica del fluido
  ForcedBGKdynamics<T, DESCRIPTOR> bulkDynamics (
    converter.getOmega(),
    instances::getBulkMomenta<T,DESCRIPTOR>()
  );
  /// Condiciones de contorno
  sOnLatticeBoundaryCondition2D<T, DESCRIPTOR> sBoundaryCondition(sLattice);
  createInterpBoundaryCondition2D<T, DESCRIPTOR> (sBoundaryCondition);

  prepareLattice(converter, sLattice, bulkDynamics, sBoundaryCondition, superGeometry);

  /// Carga de estado desde checkpoint
  int iTini = 0;
  if(useCheckpoint != 0){
    loadCheckpoint(sLattice, converter, superGeometry, iTini);
  }
  else{
    deleteOldFiles();
  }

  /// ==========================
  /// === Etapa 2: Ejecución ===
  /// ==========================
  clout << "starting simulation..." << endl;
  Timer<T> timer(converter.numTimeSteps(maxPhysT), superGeometry.getStatistics().getNvoxel());
  timer.start();

  for (int iT = iTini; iT < converter.numTimeSteps(maxPhysT); ++iT) {

    /// Aumento de Velocidad
    scaleVelocity(sLattice, converter, iT, superGeometry);

    /// Cálculo y aplicación de fuerzas del Medio Poroso
    resetForces(sLattice, converter, superGeometry);
    pMedium.applyForce(sLattice, converter);

    /// Colisión de poblaciones y propagación
    sLattice.collideAndStream();

    /// Extracción de Resultados
    getResults(sLattice, converter, iT, superGeometry, timer);
    recordReynolds(iT, lx/2., sLattice, converter);
    updatePointEvolution(iT, sLattice,converter);
    updateProfileEvolution(iT, sLattice, converter);
  }

  /// =============================
  /// === Etapa 3: Finalización ===
  /// =============================
  /// Guardado de checkpoint
  saveState(sLattice,converter);

  //Guardado de valores del punto
  recordData(polyCell, polyCellVelX, polyCellVelY, polyCellRho, "EvolucionPunto.vtk");

  //Guardado de valores del corte
  recordData(slices, polySliceVelX, polySliceVelY, polySliceRho, "EvolucionPerfil.vtk");

  timer.stop();
  timer.printSummary();
}
