// Diffussion Coefficient Ln #123
// Forcing Function Ln # 132
// Dirichlet BC Ln # 142
// Neumann BC Ln # 155
// Projection of initial data Ln #428

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;
int nrefine;
double delt,T;

template <int dim>
class ExactSol : public Function<dim>
{
public: 
  ExactSol () : Function<dim>(){}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
double  ExactSol<dim> :: value (const Point<dim>   &p,
                        const unsigned int) const
{
    double tmp, t;
    t= this ->get_time();
    //std::cout << "time=" << t << std::endl;
    tmp=std::cos(p(0)-p(1)+t);
    return tmp;
}

template <int dim>
class Step4
{
public:
  Step4 ();
  void run ();

private:
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void solve ();
  void output_results () const;
  void process_solution ();

  Triangulation<dim>   triangulation;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> stiffness_matrix;

//  ConstraintMatrix     hanging_node_constraints;

  Vector<double>       solution;
  Vector<double>       old_solution;
  Vector<double>       system_rhs;

  double	       time;
  double               ntimestep;
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
class NeunmannBC : public Function<dim>
{
public:
  NeunmannBC () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
class CoefficientK : public TensorFunction<2,dim>
{
public:
  CoefficientK() : TensorFunction<2,dim>(){}
  virtual void value_list(const std::vector<Point<dim> > &points,
                          std::vector<Tensor<2,dim> >    &values) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
  double tmp,t;
  t= this ->get_time();
  tmp=2*std::cos(p(0)-p(1)+t)-std::sin(p(0)-p(1)+t);
  return tmp;
}

template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
  double tmp,t;
  t= this ->get_time();
  if (p(1)==0)
	tmp=std::cos(p(0)+t);
  else if (p(1)==1)
  	tmp=std::cos(p(0)-1+t);
  return tmp;
}

template <int dim>
double NeunmannBC<dim>::value (const Point<dim> &p,
                                const unsigned int /*component*/) const
{
  double tmp,t;
  t= this ->get_time();
  if (p(0)==1)
	tmp=std::sin(1-p(1)+t);
  else if (p(0)==0)
  	tmp=-1*std::sin(t-p(1));
  return tmp;
}

template <int dim>
void CoefficientK<dim>::value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const
  {
    Assert (points.size() == values.size(),
            ExcDimensionMismatch (points.size(), values.size()));

    for (unsigned int p=0; p<points.size(); ++p)
      {
        values[p].clear ();

        for (unsigned int d=0; d<dim; ++d)
          values[p][d][d] = 1.;
      }
  }

template <int dim>
Step4<dim>::Step4 ()
  :
  fe (1),
  dof_handler (triangulation)
{}

template <int dim>
void Step4<dim>::make_grid ()
{
  GridGenerator::hyper_cube (triangulation, 0, 1);
  triangulation.refine_global (nrefine);

  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: "
            << triangulation.n_cells()
            << std::endl;
}

template <int dim>
void Step4<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

 // hanging_node_constraints.clear ();
 // DoFTools::make_hanging_node_constraints (dof_handler,
//                                           hanging_node_constraints);
  //hanging_node_constraints.close ();
  
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  //hanging_node_constraints.condense (dsp);
  
  system_matrix.reinit (sparsity_pattern);
  mass_matrix.reinit (sparsity_pattern);
  stiffness_matrix.reinit (sparsity_pattern);
  MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree+1),
                                      mass_matrix);
  
  solution.reinit (dof_handler.n_dofs());
  old_solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}

template <int dim>
void Step4<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(2);
  QGauss<dim-1> face_quadrature_formula(3);

  RightHandSide<dim> right_hand_side;
  NeunmannBC<dim> neunmannBC;
  const CoefficientK<dim> coefficientK;

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values         | update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<Tensor<2,dim>>    coefficient_values (n_q_points);
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);
  Vector<double>       forcing_terms;
  forcing_terms.reinit(solution.size());
  forcing_terms=0;
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  right_hand_side.set_time(time);
  neunmannBC.set_time(time);
  stiffness_matrix = 0;
  //stiffness_matrix.reinit (sparsity_pattern);

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      cell_matrix = 0;
      cell_rhs = 0;
      coefficientK.value_list (fe_values.get_quadrature_points(),
                              coefficient_values);
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (coefficient_values[q_index] *
                                   fe_values.shape_grad (i, q_index) *
                                   fe_values.shape_grad (j, q_index) *
                                   fe_values.JxW (q_index));

            cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                            right_hand_side.value (fe_values.quadrature_point (q_index)) *
                            fe_values.JxW (q_index));
          }
      for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
        if (cell->face(face_number)->at_boundary()
            &&
            (cell->face(face_number)->boundary_id() == 1))
          {
            fe_face_values.reinit (cell, face_number);

            for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
              {
                const double neunmann_value
                 // = (exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *
                   //  fe_face_values.normal_vector(q_point));
                  = -1*neunmannBC.value(fe_face_values.quadrature_point(q_point));
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  cell_rhs(i) += (neunmann_value *
                                  fe_face_values.shape_value(i,q_point) *
                                  fe_face_values.JxW(q_point));
              }
          }
      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            stiffness_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));

          forcing_terms(local_dof_indices[i]) += cell_rhs(i);
        }
    }
    forcing_terms*=delt;
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(delt,stiffness_matrix);
    mass_matrix.vmult(system_rhs,old_solution);
    system_rhs+=forcing_terms;

  //hanging_node_constraints.condense (system_matrix);
  //hanging_node_constraints.condense (system_rhs);
  std::map<types::global_dof_index,double> boundary_values;
  BoundaryValues<dim> Dirichlet_BC;
  Dirichlet_BC.set_time(time);
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            Dirichlet_BC,
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}


// @sect4{Step4::solve}

// Solving the linear system of equations is something that looks almost
// identical in most programs. In particular, it is dimension independent, so
// this function is copied verbatim from the previous example.
template <int dim>
void Step4<dim>::solve ()
{
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());
  //hanging_node_constraints.distribute (solution);
  // We have made one addition, though: since we suppress output from the
  // linear solvers, we have to print the number of iterations by hand.
  
//  std::cout << "   " << solver_control.last_step()
  //          << " CG iterations needed to obtain convergence."
    //        << std::endl;
}

template <int dim>
void Step4<dim>::output_results () const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");

  data_out.build_patches ();

  std::ofstream output (dim == 2 ?
                        "solution-2d.vtk" :
                        "solution-3d.vtk");
  data_out.write_vtk (output);
}

template <int dim>
  void Step4<dim>::process_solution()
  {
    Vector<float> difference_per_cell (triangulation.n_active_cells());
    ExactSol<dim> exactsol;
    exactsol.set_time(time);
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       exactsol,
                                       difference_per_cell,
                                       QGauss<dim>(3),
                                       VectorTools::L2_norm);
    const double L2_error = difference_per_cell.l2_norm();
    std::cout << L2_error << std::endl;
 }

// @sect4{Step4::run}

// This is the function which has the top-level control over everything. Apart
// from one line of additional output, it is the same as for the previous
// example.
template <int dim>
void Step4<dim>::run ()
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
  make_grid();
  
  typename Triangulation<dim>::cell_iterator // cell_iterator is the type of cell and endc
     cell = triangulation.begin (),
     endc = triangulation.end();
       for (; cell!=endc; ++cell){
          for (unsigned int face_number=0;
            face_number<GeometryInfo<dim>::faces_per_cell;
            ++face_number){
            if ((std::fabs(cell->face(face_number)->center()(0) - 0) < 1e-12)
                 ||
                (std::fabs(cell->face(face_number)->center()(0) - 1) < 1e-12)){
                cell->face(face_number)->set_boundary_id (1);
               }
          }
      }
  time = 0.;
  ntimestep =0;
  setup_system ();
  ExactSol<dim> exactsol;
  exactsol.set_time(time);
  VectorTools::interpolate(dof_handler,
                           exactsol,
                           old_solution);
  while (time <T){
  	time+=delt;
 	ntimestep++;
  	assemble_system ();
  	solve ();
	old_solution=solution;
  }
  output_results ();
  process_solution ();
}

int main ()
{
  std::cout << "Enter number of refinement:" << std::endl;
  std::cin >> nrefine;
  std::cout << "Enter time step:" << std::endl;
  std::cin >> delt;
  std::cout << "Enter total time:" << std::endl;
  std::cin >> T;
  deallog.depth_console (0);
  Step4<2> laplace_problem_2d;
  laplace_problem_2d.run ();

  return 0;
}
