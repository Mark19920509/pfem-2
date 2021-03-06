#ifndef PFEM2PARTICLE_H
#define PFEM2PARTICLE_H

#define PARTICLES_MOVEMENT_STEPS 1
#define MAX_PARTICLES_PER_CELL_PART 3

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q1.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/particles/particle.h>
#include <deal.II/particles/particle_handler.h>

//for parallel version
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/base/mpi.h>

#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>

#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/index_set.h>

#include <deal.II/base/utilities.h>

#include <deal.II/lac/sparsity_tools.h>

using namespace dealii;
using namespace Particles;

class pfem2Particle : public Particle<2>
{
public:
	pfem2Particle(const Point<2> & location,const Point<2> & reference_location,const types::particle_index id);
		
	void setVelocity (const Tensor<1,2> &new_velocity);
	const Tensor<1,2> & getVelocity() const;
	
private:
	Tensor<1,2> velocity;						//!< Скорость, которую переносит частица
};

class pfem2Solver
{
public:
	pfem2Solver();
	~pfem2Solver();
	
	virtual void build_grid() = 0;
	virtual void setup_system() = 0;
	virtual void assemble_system() = 0;
	virtual void solveVx(bool correction = false) = 0;
	virtual void solveVy(bool correction = false) = 0;
	virtual void solveP() = 0;
	virtual void output_results(bool predictionCorrection = false) = 0;
	
	/*!
	 * \brief Процедура первоначального "посева" частиц в ячейках сетки
	 * \param quantities Вектор количества частиц в каждом направлении (в терминах локальных координат)
	 */
	void seed_particles(const std::vector < unsigned int > & quantities);
	
	/*!
	 * \brief Коррекция скоростей частиц по скоростям в узлах сетки
	 * 
	 * Скорости частиц не сбрасываются (!). Для каждого узла сетки вычисляется изменение поля скоростей.
	 * Затем для каждой частицы по 4 узлам ячейки, в которой содержится частица, вычисляется изменение скорости (коэффициенты - значения функций формы)
	 * и посчитанное изменение добавляется к имеющейся скорости частицы.
	 */
	void correct_particles_velocities();
	
	/*!
	 * \brief "Раздача" скоростей с частиц на узлы сетки
	 * 
	 * Скорости в узлах обнуляются, после чего для каждого узла накапливается сумма скоростей от частиц (коэффициенты - значения функций формы)
	 * из ячеек, содержащих этот узел, и сумма коэффициентов. Итоговая скорость каждого узла - частное от деления первой суммы на вторую.
	 */
	void distribute_particle_velocities_to_grid();
	
	/*!
	 * \brief Перемещение частиц по известному полю скоростей в узлах
	 * 
	 * Перемещение происходит в форме 10 шагов (с шагом time_step/10). Предварительно в частицах корректируется и запоминается скорость. А затем на каждом шаге
	 * + обновляется информация о ячейке, которой принадлежит каждая частица (на первом шаге - за предварительного вычисления переносимой скорости);
	 * + вычисляется скорость частиц по скоростям в узлах сетки (на первом шаге - за предварительного вычисления переносимой скорости);
	 * + координаты частицы изменяются согласно формулам метода Эйлера.
	 */
	void move_particles();
	
	void calculate_loads(types::boundary_id patch_id, std::ofstream *out);
	
	double time,time_step;							//!< Шаг решения задачи методом конечных элементов
	int timestep_number;
	
	PETScWrappers::MPI::Vector locally_relevant_solutionVx, locally_relevant_solutionVy, locally_relevant_solutionP; //!< Вектор решения, коррекции и прогноза на текущем шаге по времени
	PETScWrappers::MPI::Vector locally_relevant_correctionVx, locally_relevant_correctionVy, 
                               locally_relevant_predictionVx, locally_relevant_predictionVy;
    PETScWrappers::MPI::Vector locally_relevant_old_solutionVx, locally_relevant_old_solutionVy, locally_relevant_old_solutionP; //!< Вектор решения на предыдущем шаге по времени (используется для вычисления разности с текущим и последующей коррекции скоростей частиц)
                               
	//MPI communicator
	MPI_Comm mpi_communicator;
	
	parallel::distributed::Triangulation<2> tria;
	ParticleHandler<2,2> particle_handler;
	FE_Q<2>  			 feVx, feVy, feP;
	DoFHandler<2>        dof_handlerVx, dof_handlerVy, dof_handlerP;
	TimerOutput			 *timer;
	
	//denote which sets of dof we own on the current processor and which we need for
    //the algorithms in this program to work
    IndexSet  locally_owned_dofsVx, locally_owned_dofsVy, locally_owned_dofsP;
    IndexSet  locally_relevant_dofsVx, locally_relevant_dofsVy, locally_relevant_dofsP;
	
	ConstraintMatrix  constraintsVx, constraintsVy;
	
    const unsigned int   dofs_per_cellVx, dofs_per_cellVy, dofs_per_cellP;

	std::vector<types::global_dof_index> local_dof_indicesVx, local_dof_indicesVy, local_dof_indicesP;
	
	//the number of total MPI processes
	const unsigned int n_mpi_processes;
	//the number of the present process
	const unsigned int this_mpi_process;
	
	//the stream which generates output one time
	ConditionalOStream pcout;
	
protected:
	void seed_particles_into_cell (const typename DoFHandler<2>::cell_iterator &cell);
	bool check_cell_for_empty_parts (const typename DoFHandler<2>::cell_iterator &cell);
		
	MappingQ1<2> mapping;
	
private:	
	std::vector < unsigned int > quantities;
	int particleCount = 0;
};

#endif // PFEM2PARTICLE_H
