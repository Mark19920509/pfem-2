#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/base/parameter_handler.h>

#include "pfem2particle.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>


using namespace dealii;

/*class InputParameter
{
public:
	static void declare_parameters (ParameterHandler &prm);
	void get_parameters (ParameterHandler &prm);
private:
	double mu, rho;
	int num_of_part_x;
	int num_of_part_y;
};

void InputParameter::declare_parameters (ParameterHandler &prm)
{
	prm.enter_subsection("Liquid characteristics");
	{
		prm.declare_entry ("Dynamic viscosity", "10.0");
		prm.declare_entry ("Density", "1.0");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Domain split");
	{
		prm.declare_entry ("Number of partitiones in the x direction", "29");
		prm.declare_entry ("Number of partitiones in the y direction", "29");
	}
	prm.leave_subsection();
}

void InputParameter::get_parameters (ParameterHandler &prm)
{
	prm.enter_subsection("Liquid characteristics");
	{
		mu = prm.get_double ("Dynamic viscosity");
		rho = prm.get_double ("Density");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Domain split");
	{
		num_of_part_x = prm.get_integer ("Number of partitiones in the x direction");
		num_of_part_y = prm.get_integer ("Number of partitiones in the y direction");
	}
	prm.leave_subsection();
}
* */

class tube : public pfem2Solver
{
public:
	tube();

	static void declare_parameters (ParameterHandler &prm);
	void get_parameters (ParameterHandler &prm);
	void build_grid ();
	void setup_system();
	void assemble_system();
	void solveVx(bool correction = false);
	void solveVy(bool correction = false);
	void solveP();
	void output_results(bool predictionCorrection = false);
	void run();
	
	QGauss<2>   quadrature_formula;
	QGauss<1>   face_quadrature_formula;
	
	FEValues<2> feVx_values, feVy_values, feP_values;
						   
	FEFaceValues<2> feVx_face_values, feVy_face_values, feP_face_values;
	
	const unsigned int   dofs_per_cellVx, dofs_per_cellVy, dofs_per_cellP;
	
	const unsigned int n_q_points;
	const unsigned int n_face_q_points;
	
	FullMatrix<double> local_matrixVx, local_matrixVy, local_matrixP;
	
	Vector<double> local_rhsVx, local_rhsVy, local_rhsP;
	
	std::vector<types::global_dof_index> local_dof_indicesVx, local_dof_indicesVy, local_dof_indicesP;
	
	double mu() const {return mu_; };
	double rho() const {return rho_; };
	
	SparsityPattern sparsity_patternVx, sparsity_patternCorrVx, sparsity_patternVy, sparsity_patternCorrVy, sparsity_patternP;
	SparseMatrix<double> system_mVx, system_mCorrVx, system_mVy, system_mCorrVy, system_mP;
	Vector<double> system_rVx, system_rVy, system_rP;
	
	ConstraintMatrix constraintsVx, constraintsVy, constraintsP, constraintsCorrVx, constraintsCorrVy;
	
private:
	double mu_, rho_;
	int num_of_part_x_, num_of_part_y_;
};

tube::tube()
	: pfem2Solver(),
	quadrature_formula(2),
	face_quadrature_formula(2),
	feVx_values (feVx, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feVy_values (feVy, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feP_values (feP, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
	feVx_face_values (feVx, face_quadrature_formula, update_values | update_quadrature_points  | update_gradients | update_normal_vectors | update_JxW_values),
	feVy_face_values (feVy, face_quadrature_formula, update_values | update_quadrature_points  | update_gradients | update_normal_vectors | update_JxW_values),
	feP_face_values (feP, face_quadrature_formula, update_values | update_quadrature_points  | update_gradients | update_normal_vectors | update_JxW_values),
	dofs_per_cellVx (feVx.dofs_per_cell),
	dofs_per_cellVy (feVy.dofs_per_cell),
	dofs_per_cellP (feP.dofs_per_cell),
	n_q_points (quadrature_formula.size()),
	n_face_q_points (face_quadrature_formula.size()),
	local_matrixVx (dofs_per_cellVx, dofs_per_cellVx),
	local_matrixVy (dofs_per_cellVy, dofs_per_cellVy),
	local_matrixP (dofs_per_cellP, dofs_per_cellP),
	local_rhsVx (dofs_per_cellVx),
	local_rhsVy (dofs_per_cellVy),
	local_rhsP (dofs_per_cellP),
	local_dof_indicesVx (dofs_per_cellVx),
	local_dof_indicesVy (dofs_per_cellVy),
	local_dof_indicesP (dofs_per_cellP)
{
	time = 0.0;
	time_step=0.01;
	timestep_number = 1;
}

void tube::declare_parameters (ParameterHandler &prm)
{
	prm.enter_subsection("Liquid characteristics");
	{
		prm.declare_entry ("Dynamic viscosity", "10.0");
		prm.declare_entry ("Density", "1.0");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Domain split");
	{
		prm.declare_entry ("Number of partitiones in the x direction", "29");
		prm.declare_entry ("Number of partitiones in the y direction", "29");
	}
	prm.leave_subsection();
}

void tube::get_parameters (ParameterHandler &prm)
{
	prm.enter_subsection("Liquid characteristics");
	{
		mu_ = prm.get_double ("Dynamic viscosity");
		rho_ = prm.get_double ("Density");
	}
	prm.leave_subsection();
	
	prm.enter_subsection("Domain split");
	{
		num_of_part_x_ = prm.get_integer ("Number of partitiones in the x direction");
		num_of_part_y_ = prm.get_integer ("Number of partitiones in the y direction");
	}
	prm.leave_subsection();
	
	std::cout << "mu = " << mu_ << "\n";
	std::cout << "rho = " << rho_ << "\n";
	std::cout << "num_of_part_x = " << num_of_part_x_ << "\n";
	std::cout << "num_of_part_y = " << num_of_part_y_ << "\n";
}

/*!
 * \brief Построение сетки
 * 
 * Используется объект tria
 */
void tube::build_grid ()
{
  TimerOutput::Scope timer_section(*timer, "Mesh construction");
  
  const Point<2> bottom_left = Point<2> (0,0);
  const Point<2> top_right = Point<2> (2,2);

  std::vector< unsigned int > repetitions {(unsigned int)(num_of_part_x_), (unsigned int)(num_of_part_y_)}; 

  GridGenerator::subdivided_hyper_rectangle(tria,repetitions,bottom_left,top_right, true);
  
  return;
  
  std::cout << "Grid has " << tria.n_cells(tria.n_levels()-1) << " cells" << std::endl;
  
  GridOut grid_out;

  std::ofstream out ("tube.eps");
  grid_out.write_eps (tria, out);
  std::cout << "Grid written to EPS" << std::endl;
  
  std::ofstream out2 ("tube.vtk");
  grid_out.write_vtk (tria, out2);  
  std::cout << "Grid written to VTK" << std::endl;
}

void tube::setup_system()
{
	TimerOutput::Scope timer_section(*timer, "System setup");

	dof_handlerVx.distribute_dofs (feVx);
	std::cout << "Number of degrees of freedom Vx: "
			  << dof_handlerVx.n_dofs()
			  << std::endl;
			  
	dof_handlerVy.distribute_dofs (feVy);
	std::cout << "Number of degrees of freedom Vy: "
			  << dof_handlerVy.n_dofs()
			  << std::endl;
			  
	dof_handlerP.distribute_dofs (feP);
	std::cout << "Number of degrees of freedom P: "
			  << dof_handlerP.n_dofs()
			  << std::endl;

	//Vx	  
	DynamicSparsityPattern dspVx(dof_handlerVx.n_dofs());
	
	solutionVx.reinit (dof_handlerVx.n_dofs());
	predictionVx.reinit (dof_handlerVx.n_dofs());
	correctionVx.reinit (dof_handlerVx.n_dofs());
	old_solutionVx.reinit (dof_handlerVx.n_dofs());
    system_rVx.reinit (dof_handlerVx.n_dofs());
    
    constraintsVx.clear();
    DoFTools::make_hanging_node_constraints(dof_handlerVx, constraintsVx);
    VectorTools::interpolate_boundary_values(dof_handlerVx, 2, Functions::ConstantFunction<2>(0.0), constraintsVx);
    VectorTools::interpolate_boundary_values(dof_handlerVx, 3, Functions::ConstantFunction<2>(0.0), constraintsVx);
    constraintsVx.close();
    
    constraintsCorrVx.clear();
    DoFTools::make_hanging_node_constraints(dof_handlerVx, constraintsCorrVx);
    VectorTools::interpolate_boundary_values(dof_handlerVx, 2, Functions::ConstantFunction<2>(0.0), constraintsCorrVx);
    VectorTools::interpolate_boundary_values(dof_handlerVx, 3, Functions::ConstantFunction<2>(0.0), constraintsCorrVx);
    constraintsCorrVx.close();

	DoFTools::make_sparsity_pattern (dof_handlerVx, dspVx, constraintsVx, false);
	sparsity_patternVx.copy_from(dspVx);
	
	system_mVx.reinit (sparsity_patternVx);
	
	DoFTools::make_sparsity_pattern (dof_handlerVx, dspVx, constraintsCorrVx, false);
	sparsity_patternCorrVx.copy_from(dspVx);
	
	system_mCorrVx.reinit (sparsity_patternCorrVx);
	
	//assembling the system matrix (prediction Vx)
	system_mVx=0.0;
	
	{
		DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
						
		feVx_values.reinit (cell);
		local_matrixVx = 0.0;
			
		for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
			for (unsigned int i=0; i<dofs_per_cellVx; ++i) {						
				for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
					local_matrixVx(i,j) += rho() * feVx_values.shape_value (i,q_index) * feVx_values.shape_value (j,q_index) * feVx_values.JxW(q_index);
					//implicit account for tau_ij
					local_matrixVx(i,j) += mu() * time_step * (feVx_values.shape_grad (i,q_index)[1] * feVx_values.shape_grad (j,q_index)[1] + 
					                       4.0/3.0 * feVx_values.shape_grad (i,q_index)[0] * feVx_values.shape_grad (j,q_index)[0]) * feVx_values.JxW (q_index);
				}//j
			}//i
		}//q_index
			
		for (; cell!=endc; ++cell) {
			feVx_values.reinit (cell);
				
			cell->get_dof_indices (local_dof_indicesVx);
			constraintsVx.distribute_local_to_global(local_matrixVx, local_dof_indicesVx, system_mVx);
		}//cell
	}// end of assembling the system matrix (prediction Vx)
	
	//assembling the system matrix (correction Vx)
	system_mCorrVx = 0.0;
	
	{		
		DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
			
		feVx_values.reinit (cell);
		local_matrixVx = 0.0;
		
		for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
			for (unsigned int i=0; i<dofs_per_cellVx; ++i) {					
				for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
					local_matrixVx(i,j) += feVx_values.shape_value (i,q_index) * feVx_values.shape_value (j,q_index) * feVx_values.JxW(q_index);
				}//j
			}//i
		}//q_index
      
		for (; cell!=endc; ++cell) {
			feVx_values.reinit (cell);
			cell->get_dof_indices (local_dof_indicesVx);
			constraintsCorrVx.distribute_local_to_global(local_matrixVx, local_dof_indicesVx, system_mCorrVx);

		}//cell
			
	}//end of assembling the system matrix (correction Vx)
    
    //Vy
    DynamicSparsityPattern dspVy(dof_handlerVy.n_dofs());
		
	solutionVy.reinit (dof_handlerVy.n_dofs());
	predictionVy.reinit (dof_handlerVy.n_dofs());
	correctionVy.reinit (dof_handlerVy.n_dofs());
	old_solutionVy.reinit (dof_handlerVy.n_dofs());
    system_rVy.reinit (dof_handlerVy.n_dofs());
    
    constraintsVy.clear();
    DoFTools::make_hanging_node_constraints(dof_handlerVy, constraintsVy);
    VectorTools::interpolate_boundary_values(dof_handlerVy, 2, Functions::ConstantFunction<2>(0.0), constraintsVy);
    VectorTools::interpolate_boundary_values(dof_handlerVy, 3, Functions::ConstantFunction<2>(0.0), constraintsVy);
    constraintsVy.close();
    
    constraintsCorrVy.clear();
    DoFTools::make_hanging_node_constraints(dof_handlerVy, constraintsCorrVy);
    VectorTools::interpolate_boundary_values(dof_handlerVy, 2, Functions::ConstantFunction<2>(0.0), constraintsCorrVy);
    VectorTools::interpolate_boundary_values(dof_handlerVy, 3, Functions::ConstantFunction<2>(0.0), constraintsCorrVy);
    constraintsCorrVy.close();
	
	DoFTools::make_sparsity_pattern (dof_handlerVy, dspVy, constraintsVy, false);
	sparsity_patternVy.copy_from(dspVy);
	
	system_mVy.reinit (sparsity_patternVy);
	
	DoFTools::make_sparsity_pattern (dof_handlerVy, dspVy, constraintsCorrVy, false);
	sparsity_patternCorrVy.copy_from(dspVy);
	
	system_mCorrVy.reinit (sparsity_patternCorrVy);
	
	//assembling the system matrix (prediction Vy)
	system_mVy=0.0;
	
	{
		DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();
						
		feVy_values.reinit (cell);
		local_matrixVy = 0.0;

		for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
			for (unsigned int i=0; i<dofs_per_cellVy; ++i) {		
				for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
					local_matrixVy(i,j) += rho() * feVy_values.shape_value (i,q_index) * feVy_values.shape_value (j,q_index) * feVy_values.JxW(q_index);
					//implicit account for tau_ij
					local_matrixVy(i,j) += mu() * time_step * (feVy_values.shape_grad (i,q_index)[0] * feVy_values.shape_grad (j,q_index)[0] + 
										   4.0/3.0 * feVy_values.shape_grad (i,q_index)[1] * feVy_values.shape_grad (j,q_index)[1]) * feVy_values.JxW (q_index);																			
				}//j
			}//i
		}//q_index
		
		for (; cell!=endc; ++cell) {
			feVy_values.reinit (cell);
			cell->get_dof_indices (local_dof_indicesVy);
			constraintsVy.distribute_local_to_global(local_matrixVy, local_dof_indicesVy, system_mVy);
		}//cell
	}//end of assembling the system matrix (prediction Vy)
	
	//assembling the system matrix (correction Vy)
	system_mCorrVy = 0.0;
	
	{	
		DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();
		
		feVy_values.reinit (cell);
		local_matrixVy = 0.0;
		
		for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
			for (unsigned int i=0; i<dofs_per_cellVy; ++i) {
				for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
					local_matrixVy(i,j) += feVy_values.shape_value (i,q_index) * feVy_values.shape_value (j,q_index) * feVy_values.JxW(q_index);
				}//j
			}//i
		}//q_index
			
		for (; cell!=endc; ++cell) {
			feVy_values.reinit (cell);
      
			cell->get_dof_indices (local_dof_indicesVy);
			constraintsCorrVy.distribute_local_to_global(local_matrixVy, local_dof_indicesVy, system_mCorrVy);

		}//cell
				
	}//end of assembling the system matrix (correction Vy)
	
	//P
    DynamicSparsityPattern dspP(dof_handlerP.n_dofs());

	solutionP.reinit (dof_handlerP.n_dofs());
	old_solutionP.reinit (dof_handlerP.n_dofs());
    system_rP.reinit (dof_handlerP.n_dofs());
    
    constraintsP.clear();
    DoFTools::make_hanging_node_constraints(dof_handlerP, constraintsP);
    VectorTools::interpolate_boundary_values(dof_handlerP, 1, Functions::ConstantFunction<2>(0.0), constraintsP);
    constraintsP.close();

	DoFTools::make_sparsity_pattern (dof_handlerP, dspP, constraintsP, false);
	sparsity_patternP.copy_from(dspP);
	
	system_mP.reinit (sparsity_patternP);
	
	//assembling the system matrix (P)
	system_mP=0.0;
	
	{
		DoFHandler<2>::active_cell_iterator cell = dof_handlerP.begin_active();
		DoFHandler<2>::active_cell_iterator endc = dof_handlerP.end();
		
		feP_values.reinit (cell);
		local_matrixP = 0.0;
					
		for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
			for (unsigned int i=0; i<dofs_per_cellP; ++i) {
				for (unsigned int j=0; j<dofs_per_cellP; ++j) {
					local_matrixP(i,j) += feP_values.shape_grad (i,q_index) * feP_values.shape_grad (j,q_index) * feP_values.JxW(q_index);						
				}//j
			}//i
		}//q_index
			
		for (; cell!=endc; ++cell) {
			feP_values.reinit (cell);

			cell->get_dof_indices (local_dof_indicesP);
			constraintsP.distribute_local_to_global(local_matrixP, local_dof_indicesP, system_mP);
		}//cell
	}//end of assembling the system matrix (P)
    
    return;
    
    DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active(), endc = dof_handlerVx.end();
	std::ofstream vertices("vertices.txt");
	for (; cell!=endc; ++cell) {
		for (unsigned int i=0; i < 4; ++i){
			vertices << "DoF no. " << cell->vertex_dof_index(i,0) << " is located at " << cell->vertex(i) << std::endl;
		}
	}
	
	vertices.close();
}

void tube::assemble_system()
{
	TimerOutput::Scope timer_section(*timer, "FEM step");
	
	old_solutionVx = solutionVx; 
	old_solutionVy = solutionVy;
	old_solutionP = solutionP;
	
	for(int nOuterCorr = 0; nOuterCorr < 1; ++nOuterCorr){
				
		//Vx
		system_rVx=0.0;
		{
			DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
			DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
			
			int number = 0;
			
			for (; cell!=endc; ++cell,++number) {
				feVx_values.reinit (cell);
				feVy_values.reinit (cell);
				feP_values.reinit (cell);
				local_rhsVx = 0.0;
			
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVx; ++i) {
						const Tensor<0,2> Ni_vel = feVx_values.shape_value (i,q_index);
						const Tensor<1,2> Ni_vel_grad = feVx_values.shape_grad (i,q_index);
						
						for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
							const Tensor<0,2> Nj_vel = feVx_values.shape_value (j,q_index);
							const Tensor<1,2> Nj_vel_grad = feVx_values.shape_grad (j,q_index);
							const Tensor<1,2> Nj_p_grad = feP_values.shape_grad (j,q_index);
							
							//explicit account for tau_ij
							//local_rhsVx(i) -= mu * time_step * (Ni_vel_grad[1] * Nj_vel_grad[1] + 4.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[0]) * old_solutionVx(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
							local_rhsVx(i) -= mu() * time_step * (Ni_vel_grad[1] * Nj_vel_grad[0] - 2.0/3.0 * Ni_vel_grad[0] * Nj_vel_grad[1]) * old_solutionVy(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
							
							local_rhsVx(i) += rho() * Nj_vel * Ni_vel * old_solutionVx(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
							local_rhsVx(i) -= time_step * Ni_vel * Nj_p_grad[0] * old_solutionP(cell->vertex_dof_index(j,0)) * feVx_values.JxW (q_index);
						}//j
					}//i
				}//q_index
				
				for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
					if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 0 || cell->face(face_number)->boundary_id() == 1)){
						feVx_face_values.reinit (cell, face_number);
						
						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
							double duydy = 0.0;
							for (unsigned int i=0; i<dofs_per_cellVy; ++i)
								duydy += feVx_face_values.shape_grad(i,q_point)[1] * old_solutionVy(cell->vertex_dof_index(i,0));
						
							for (unsigned int i=0; i<dofs_per_cellVy; ++i)
								local_rhsVx(i) += mu() * time_step * feVx_face_values.shape_value(i,q_point) * (-2.0/3.0) * duydy *
									feVx_face_values.normal_vector(q_point)[0] * feVx_face_values.JxW(q_point);
						}
					}
		  
				cell->get_dof_indices (local_dof_indicesVx);
				constraintsVx.distribute_local_to_global(local_rhsVx, local_dof_indicesVx, system_rVx);
			}//cell
		}//Vx

		solveVx ();
		
		//Vy
		system_rVy=0.0;
		
		{
			DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
			DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();
				
			for (; cell!=endc; ++cell) {
				feVx_values.reinit (cell);
				feVy_values.reinit (cell);
				feP_values.reinit (cell);
				local_rhsVy = 0.0;

				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellVy; ++i) {
						const Tensor<0,2> Ni_vel = feVy_values.shape_value (i,q_index);
						const Tensor<1,2> Ni_vel_grad = feVy_values.shape_grad (i,q_index);
		
						for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
							const Tensor<0,2> Nj_vel = feVy_values.shape_value (j,q_index);
							const Tensor<1,2> Nj_vel_grad = feVy_values.shape_grad (j,q_index);
							const Tensor<1,2> Nj_p_grad = feP_values.shape_grad (j,q_index);					
													
							//explicit account for tau_ij
							local_rhsVy(i) -= mu() * time_step * (Ni_vel_grad[0] * Nj_vel_grad[1] - 2.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[0]) * old_solutionVx(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index);
							//local_rhsVy(i) -= mu * time_step * (Ni_vel_grad[0] * Nj_vel_grad[0] + 4.0/3.0 * Ni_vel_grad[1] * Nj_vel_grad[1]) * old_solutionVy(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index);

							local_rhsVy(i) += rho() * Nj_vel * Ni_vel * old_solutionVy(cell->vertex_dof_index(j,0)) *  feVy_values.JxW (q_index); 
							local_rhsVy(i) -= time_step * Ni_vel * Nj_p_grad[1] * old_solutionP(cell->vertex_dof_index(j,0)) * feVy_values.JxW (q_index);
						}//j
					}//i
				}//q_index
				
				for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
					if (cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id() == 0 || cell->face(face_number)->boundary_id() == 1)){
						feVy_face_values.reinit (cell, face_number);
						
						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
							double duxdy = 0.0;
							for (unsigned int i=0; i<dofs_per_cellVy; ++i)
								duxdy += feVy_face_values.shape_grad(i,q_point)[1] * old_solutionVx(cell->vertex_dof_index(i,0));
						
							for (unsigned int i=0; i<dofs_per_cellVy; ++i)
								local_rhsVy(i) += mu() * time_step * feVy_face_values.shape_value(i,q_point) * duxdy *
									feVy_face_values.normal_vector(q_point)[0] * feVy_face_values.JxW(q_point);
						}
					}
		  
				cell->get_dof_indices (local_dof_indicesVy);
				constraintsVy.distribute_local_to_global(local_rhsVy, local_dof_indicesVy, system_rVy);
			}//cell
		}//Vy
		
		solveVy ();

	//P	
	for (int n_cor=0; n_cor<1; ++n_cor){
		system_rP=0.0;
		
		{
			DoFHandler<2>::active_cell_iterator cell = dof_handlerP.begin_active();
			DoFHandler<2>::active_cell_iterator endc = dof_handlerP.end();
		
			for (; cell!=endc; ++cell) {
				feVx_values.reinit (cell);
				feVy_values.reinit (cell);
				feP_values.reinit (cell);
				local_rhsP = 0.0;
					
				for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
					for (unsigned int i=0; i<dofs_per_cellP; ++i) {
						const Tensor<1,2> Nidx_pres = feP_values.shape_grad (i,q_index);
						
						for (unsigned int j=0; j<dofs_per_cellP; ++j) {
							const Tensor<0,2> Nj_vel = feVx_values.shape_value (j,q_index);
							const Tensor<1,2> Njdx_pres = feP_values.shape_grad (j,q_index);
											
							local_rhsP(i) += rho() / time_step * (predictionVx(cell->vertex_dof_index(j,0)) * Nidx_pres[0] + 
							                   predictionVy(cell->vertex_dof_index(j,0)) * Nidx_pres[1]) * Nj_vel * feP_values.JxW (q_index);
							local_rhsP(i) += Nidx_pres * Njdx_pres * old_solutionP(cell->vertex_dof_index(j,0)) * feP_values.JxW(q_index);
						}//j
					}//i
				}//q_index


				for (unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
					if (cell->face(face_number)->at_boundary()){
						feP_face_values.reinit (cell, face_number);

						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point){
							double dp_dn_q_point = 0.0;
							double Vx_q_point_value = 0.0;

							for (unsigned int i=0; i<dofs_per_cellP; ++i){
								dp_dn_q_point += feP_face_values.shape_grad(i,q_point) * feP_face_values.normal_vector(q_point) * old_solutionP(cell->vertex_dof_index(i,0));
							
								if(cell->face(face_number)->boundary_id() == 0 || cell->face(face_number)->boundary_id() == 1)
									Vx_q_point_value += feP_face_values.shape_value(i,q_point) * predictionVx(cell->vertex_dof_index(i,0));								
							}
						
							for (unsigned int i=0; i<dofs_per_cellP; ++i){
								local_rhsP(i) -= feP_face_values.shape_value(i,q_point) * dp_dn_q_point * feP_face_values.JxW(q_point);
								
								if(cell->face(face_number)->boundary_id() == 0 || cell->face(face_number)->boundary_id() == 1)
									local_rhsP(i) -= rho() / time_step * feP_face_values.shape_value(i,q_point) * Vx_q_point_value *
										feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);

								if(cell->face(face_number)->boundary_id() == 0)
									local_rhsP(i) += feP_face_values.shape_value(i,q_point) * (-10.0) * feP_face_values.normal_vector(q_point)[0] * feP_face_values.JxW(q_point);

							}
						}
					}

					cell->get_dof_indices (local_dof_indicesP);
					constraintsP.distribute_local_to_global(local_rhsP, local_dof_indicesP, system_rP);
				}//cell
			}//P
		
			solveP ();
	
			//Vx correction
			{
				system_rVx = 0.0;
				
				DoFHandler<2>::active_cell_iterator cell = dof_handlerVx.begin_active();
				DoFHandler<2>::active_cell_iterator endc = dof_handlerVx.end();
		
				for (; cell!=endc; ++cell) {
					feVx_values.reinit (cell);
					feVy_values.reinit (cell);
					feP_values.reinit (cell);
					local_rhsVx = 0.0;
		
					for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
						for (unsigned int i=0; i<dofs_per_cellVx; ++i) {
							const Tensor<0,2> Ni_vel = feVx_values.shape_value (i,q_index);
					
							for (unsigned int j=0; j<dofs_per_cellVx; ++j) {
								const Tensor<1,2> Nj_p_grad = feP_values.shape_grad (j,q_index);

								local_rhsVx(i) -= time_step/rho() * Ni_vel * Nj_p_grad[0] * (solutionP(cell->vertex_dof_index(j,0)) - old_solutionP(cell->vertex_dof_index(j,0))) * feVx_values.JxW (q_index);
							}//j
						}//i
					}//q_index
      
					cell->get_dof_indices (local_dof_indicesVx);
					constraintsCorrVx.distribute_local_to_global( local_rhsVx, local_dof_indicesVx, system_rVx);

				}//cell
			
			}//correction Vx
		
			solveVx (true);
		
			//Vy correction
			{
				system_rVy = 0.0;
			
				DoFHandler<2>::active_cell_iterator cell = dof_handlerVy.begin_active();
				DoFHandler<2>::active_cell_iterator endc = dof_handlerVy.end();
		
				for (; cell!=endc; ++cell) {
					feVx_values.reinit (cell);
					feVy_values.reinit (cell);
					feP_values.reinit (cell);
					local_rhsVy = 0.0;
		
					for (unsigned int q_index=0; q_index<n_q_points; ++q_index) {
						for (unsigned int i=0; i<dofs_per_cellVy; ++i) {
							const Tensor<0,2> Ni_vel = feVy_values.shape_value (i,q_index);
					
							for (unsigned int j=0; j<dofs_per_cellVy; ++j) {
								const Tensor<1,2> Nj_p_grad = feP_values.shape_grad (j,q_index);

								local_rhsVy(i) -= time_step/rho() * Ni_vel * Nj_p_grad[1] * (solutionP(cell->vertex_dof_index(j,0)) - old_solutionP(cell->vertex_dof_index(j,0))) * feVy_values.JxW (q_index);
							}//j
						}//i
					}//q_index
      
					cell->get_dof_indices (local_dof_indicesVy);
					constraintsCorrVy.distribute_local_to_global(local_rhsVy, local_dof_indicesVy, system_rVy);

				}//cell
				
			}//Vy
			
			solveVy (true);
		
		
			solutionVx = predictionVx;
			solutionVx += correctionVx;
			solutionVy = predictionVy;
			solutionVy += correctionVy;
		
			old_solutionP = solutionP;
		}//n_cor
	}//nOuterCorr
}


/*!
 * \brief Решение системы линейных алгебраических уравнений для МКЭ
 */
void tube::solveVx(bool correction)
{
	SolverControl solver_control (10000, 1e-13);
	SolverBicgstab<> solver (solver_control);
	PreconditionJacobi<> preconditioner;
	
	preconditioner.initialize(system_mVx, 1.0);
	if(correction) solver.solve (system_mCorrVx, correctionVx, system_rVx, preconditioner);
	else solver.solve (system_mVx, predictionVx, system_rVx, preconditioner);

	if(correction) constraintsCorrVx.distribute(correctionVx);
	else constraintsCorrVx.distribute(predictionVx);

    if(solver_control.last_check() == SolverControl::success)
		std::cout << "Solver for Vx converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else std::cout << "Solver for Vx failed to converge" << std::endl;
}

void tube::solveVy(bool correction)
{
	SolverControl solver_control (10000, 1e-13);
	SolverBicgstab<> solver (solver_control);
	PreconditionJacobi<> preconditioner;
	
	preconditioner.initialize(system_mVy, 1.0);
	if(correction) solver.solve (system_mCorrVy, correctionVy, system_rVy, preconditioner);
	else solver.solve (system_mVy, predictionVy, system_rVy, preconditioner);

	if(correction) constraintsCorrVy.distribute(correctionVy);
	else constraintsCorrVy.distribute(predictionVy);

    if(solver_control.last_check() == SolverControl::success)
		std::cout << "Solver for Vy converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else std::cout << "Solver for Vy failed to converge" << std::endl;
}

void tube::solveP()
{
	SolverControl solver_control (10000, 1e-13);
	SolverBicgstab<> solver (solver_control);

	PreconditionSSOR<> preconditioner;
	
	preconditioner.initialize(system_mP, 1.0);
	solver.solve (system_mP, solutionP, system_rP,
                  preconditioner);
              
	constraintsP.distribute(solutionP);

    if(solver_control.last_check() == SolverControl::success)
		std::cout << "Solver for P converged with residual=" << solver_control.last_value() << ", no. of iterations=" << solver_control.last_step() << std::endl;
	else std::cout << "Solver for P failed to converge" << std::endl;
}

/*!
 * \brief Вывод результатов в формате VTK
 */
void tube::output_results(bool predictionCorrection) 
{
	TimerOutput::Scope timer_section(*timer, "Results output");
	
	/*const std::string filenameVx =  "resultsVx-" + Utilities::int_to_string (timestep_number, 2) +	".txt";
	std::ofstream rawResults (filenameVx.c_str());
	for (unsigned int i=0; i<dof_handlerVx.n_dofs(); ++i){
		rawResults << "DoF no. " << i << ", Vx=" << solutionVx(i) << std::endl;
	}
	rawResults.close();*/
	
	DataOut<2> data_out;

	data_out.attach_dof_handler (dof_handlerVx);
	data_out.add_data_vector (solutionVx, "Vx");
	data_out.add_data_vector (solutionVy, "Vy");
	data_out.add_data_vector (solutionP, "P");
	
	if(predictionCorrection){
		data_out.add_data_vector (predictionVx, "predVx");
		data_out.add_data_vector (predictionVy, "predVy");
		data_out.add_data_vector (correctionVx, "corVx");
		data_out.add_data_vector (correctionVy, "corVy");
	}
	
	data_out.build_patches ();

	const std::string filename =  "solution-" + Utilities::int_to_string (timestep_number, 2) +	".vtk";
	std::ofstream output (filename.c_str());
	data_out.write_vtk (output);
	
	//вывод частиц
	const std::string filename2 =  "particles-" + Utilities::int_to_string (timestep_number, 2) + ".vtk";
	std::ofstream output2 (filename2.c_str());
	output2 << "# vtk DataFile Version 3.0" << std::endl;
	output2 << "Unstructured Grid Example" << std::endl;
	output2 << "ASCII" << std::endl;
	output2 << std::endl;
	output2 << "DATASET UNSTRUCTURED_GRID" << std::endl;
	output2 << "POINTS " << particle_handler.n_global_particles() << " float" << std::endl;
	for(ParticleIterator<2> particleIndex = particle_handler.begin(); 
		                                   particleIndex != particle_handler.end(); ++particleIndex){
		output2 << particleIndex->get_location() << " 0" << std::endl;
	}
	
	output2 << std::endl;
	
	output2 << "CELLS " << particle_handler.n_global_particles() << " " << 2 * particle_handler.n_global_particles() << std::endl;
	for (unsigned int i=0; i< particle_handler.n_global_particles(); ++i){
		output2 << "1 " << i << std::endl; 
	}
	
	output2 << std::endl;
	
	output2 << "CELL_TYPES " << particle_handler.n_global_particles() << std::endl;
	for (unsigned int i=0; i< particle_handler.n_global_particles(); ++i){
		output2 << "1 "; 
	}	
	output2 << std::endl;
	
	output2 << std::endl;
	
	output2 << "POINT_DATA " << particle_handler.n_global_particles() << std::endl;
	output2 << "VECTORS velocity float" << std::endl;
	for(ParticleIterator<2> particleIndex = particle_handler.begin(); 
		                                   particleIndex != particle_handler.end(); ++particleIndex){
		output2 << velocity_x[particleIndex->get_id()] << " " << velocity_y[particleIndex->get_id()] << " 0" << std::endl;
	}
}

/*!
 * \brief Основная процедура программы
 * 
 * Подготовительные операции, цикл по времени, вызов вывода результатов
 */
void tube::run()
{	
	timer = new TimerOutput(std::cout, TimerOutput::summary, TimerOutput::wall_times);
	
	build_grid();
	setup_system();
	seed_particles({2, 2});
	
	solutionVx=0.0;
	solutionVy=0.0;
	solutionP=0.0;
	
	//удаление старых файлов VTK (специфическая команда Linux!!!)
	system("rm solution-*.vtk");
	system("rm particles-*.vtk");

//	std::ofstream os("force.csv");

	for (; time<=100; time+=time_step, ++timestep_number) {
		std::cout << std::endl << "Time step " << timestep_number << " at t=" << time << std::endl;
		
		//correct_particles_velocities();
		//move_particles();
		//distribute_particle_velocities_to_grid();
		
		assemble_system();
		//if((timestep_number - 1) % 1 == 0) 
			output_results(true);
		
		//calculate_loads(3, &os);
		
		timer->print_summary();
	}//time
	
//	os.close();
	
	delete timer;
}

int main (int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
	
	ParameterHandler prm;
	tube tubeproblem;

	tubeproblem.declare_parameters (prm);
	prm.parse_input ("input_data.prm");	
	
	prm.print_parameters (std::cout, ParameterHandler::Text);
	// get parameters into the program
	std::cout << "\n\n" << "Getting parameters:" << std::endl;
	
	tubeproblem.get_parameters (prm);
	
	//std::cout << "mu = " << tubeproblem.mu() << "\n"; 
	//std::cout << "rho = " << tubeproblem.rho() << "\n";
	//std::cout << "__________________________________________________________________________________\n";
	//std::cout << "__________________________________________________________________________________\n";
	
	tubeproblem.run ();
  
	return 0;
}
