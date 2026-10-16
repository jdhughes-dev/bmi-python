"""The Basic Model Interface (BMI) Python specification.

This language specification is derived from the Scientific Interface
Definition Language (SIDL) file `bmi.sidl <https://github.com/csdms/bmi>`_.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


class BmiExtension(ABC):
    """Base class for the objects returned by :func:`Bmi.get_extension`."""


class BmiPointers(BmiExtension):
    """Extension object for ``"bmi:pointers"``."""

    @abstractmethod
    def get_value_ptr(self, index: int) -> NDArray[Any]:
        """Get a reference to values of the given variable.

        This is a getter for the model, used to access the model's
        current state. It returns a reference to a model variable,
        with the return type, size and rank dependent on the variable.
        A variable in the ``"bmi:pointers:stable"`` set keeps its
        reference from :func:`Bmi.initialize` to :func:`Bmi.finalize`;
        one in ``"bmi:pointers:transient"`` must be fetched again after
        any call that changes the model's state.

        Parameters
        ----------
        index : int
            The index of the variable, from :func:`Bmi.get_var_index`.

        Returns
        -------
        array_like
            A reference to a model variable.
        """
        ...


class Bmi(ABC):
    """Base class for implementations of a Python-based BMI."""

    @abstractmethod
    def initialize(
        self, config_file: str, requested_extensions: Sequence[str] = ()
    ) -> tuple[str, ...]:
        """Perform startup tasks for the model and negotiate extensions.

        Perform all tasks that take place before entering the model's time
        loop, including opening files and initializing the model state. Model
        inputs are read from a text-based configuration file, specified by
        `config_file`.

        Parameters
        ----------
        config_file : str
            The path to the model configuration file.
        requested_extensions : sequence of str, optional
            Names of the extensions the caller would like to use.

        Returns
        -------
        tuple of str
            The requested extensions that the model supports.

        Notes
        -----
        Models should be refactored, if necessary, to use a
        configuration file. CSDMS does not impose any constraint on
        how configuration files are formatted, although YAML is
        recommended. A template of a model's configuration file
        with placeholder values is used by the BMI.
        """
        ...

    @abstractmethod
    def get_extension(self, extension_name: str) -> BmiExtension:
        """Get the object that implements the named extension.

        Parameters
        ----------
        extension_name : str
            The name of an extension the model reported as supported by
            :func:`initialize`.

        Returns
        -------
        BmiExtension
            The extension object.
        """
        ...

    @abstractmethod
    def update(self) -> None:
        """Advance model state by one time step.

        Perform all tasks that take place within one pass through the model's
        time loop. This typically includes incrementing all of the model's
        state variables. If the model's state variables don't change in time,
        then they can be computed by the :func:`initialize` method and this
        method can return with no action.
        """
        ...

    @abstractmethod
    def update_until(self, time: float) -> None:
        """Advance model state until the given time.

        Parameters
        ----------
        time : float
            A model time later than the current model time.
        """
        ...

    @abstractmethod
    def finalize(self) -> None:
        """Perform tear-down tasks for the model.

        Perform all tasks that take place after exiting the model's time
        loop. This typically includes deallocating memory, closing files and
        printing reports.
        """
        ...

    @abstractmethod
    def get_component_name(self) -> str:
        """Name of the component.

        Returns
        -------
        str
            The name of the component.
        """
        ...

    @abstractmethod
    def get_varset_member_count(self, set_name: str) -> int:
        """Count the variables in a variable set.

        Parameters
        ----------
        set_name : str
            The name of the variable set, for example ``"input"`` or
            ``"output"``.

        Returns
        -------
        int
          The number of variables in the set.
        """
        ...

    @abstractmethod
    def get_varset_members(self, set_name: str) -> tuple[str, ...]:
        """Get the variables in a variable set.

        Variable names must be CSDMS Standard Names, also known as *long
        variable names*.

        Parameters
        ----------
        set_name : str
            The name of the variable set, for example ``"input"`` or
            ``"output"``.

        Returns
        -------
        tuple of str
            The variables in the set.

        Notes
        -----
        Standard Names enable the CSDMS framework to determine whether
        an input variable in one model is equivalent to, or compatible
        with, an output variable in another model. This allows the
        framework to automatically connect components.

        Standard Names do not have to be used within the model.
        """
        ...

    @abstractmethod
    def get_var_index(self, name: str) -> int:
        """Get the index of the given variable.

        The index identifies the variable in every other variable function.
        Indices need not be ordered or contiguous.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        int
            The variable index.
        """
        ...

    @abstractmethod
    def get_var_grid(self, index: int) -> int:
        """Get grid identifier for the given variable.

        Parameters
        ----------
        index : int
            The index of the variable, from :func:`Bmi.get_var_index`.

        Returns
        -------
        int
          The grid identifier.
        """
        ...

    @abstractmethod
    def get_var_type(self, index: int) -> str:
        """Get data type of the given variable.

        Parameters
        ----------
        index : int
            The index of the variable, from :func:`Bmi.get_var_index`.

        Returns
        -------
        str
            The Python variable type; e.g., ``str``, ``int``, ``float``.
        """
        ...

    @abstractmethod
    def get_var_units(self, index: int) -> str:
        """Get units of the given variable.

        Standard unit names, in lower case, should be used, such as
        ``meters`` or ``seconds``. Standard abbreviations, like ``m`` for
        meters, are also supported. For variables with compound units,
        each unit name is separated by a single space, with exponents
        other than 1 placed immediately after the name, as in ``m s-1``
        for velocity, ``W m-2`` for an energy flux, or ``km2`` for an
        area.

        Parameters
        ----------
        index : int
            The index of the variable, from :func:`Bmi.get_var_index`.

        Returns
        -------
        str
            The variable units.

        Notes
        -----
        CSDMS uses the `UDUNITS`_ standard from Unidata.

        .. _UDUNITS: http://www.unidata.ucar.edu/software/udunits
        """
        ...

    @abstractmethod
    def get_var_itemsize(self, index: int) -> int:
        """Get memory use for each array element in bytes.

        Parameters
        ----------
        index : int
            The index of the variable, from :func:`Bmi.get_var_index`.

        Returns
        -------
        int
            Item size in bytes.
        """
        ...

    @abstractmethod
    def get_var_nbytes(self, index: int) -> int:
        """Get size, in bytes, of the given variable.

        Parameters
        ----------
        index : int
            The index of the variable, from :func:`Bmi.get_var_index`.

        Returns
        -------
        int
            The size of the variable, counted in bytes.
        """
        ...

    @abstractmethod
    def get_var_location(self, index: int) -> str:
        """Get the grid element type that the a given variable is defined on.

        The grid topology can be composed of *nodes*, *edges*, and *faces*.

        *node*
            A point that has a coordinate pair or triplet: the most
            basic element of the topology.

        *edge*
            A line or curve bounded by two *nodes*.

        *face*
            A plane or surface enclosed by a set of edges. In a 2D
            horizontal application one may consider the word "polygon",
            but in the hierarchy of elements the word "face" is most common.

        Parameters
        ----------
        index : int
            The index of the variable, from :func:`Bmi.get_var_index`.

        Returns
        -------
        str
            The grid location on which the variable is defined. Must be one of
            `"node"`, `"edge"`, or `"face"`.

        Notes
        -----
        CSDMS uses the `ugrid conventions`_ to define unstructured grids.

        .. _ugrid conventions: http://ugrid-conventions.github.io/ugrid-conventions
        """
        ...

    @abstractmethod
    def get_current_time(self) -> float:
        """Return the current time of the model.

        Returns
        -------
        float
            The current model time.
        """
        ...

    @abstractmethod
    def get_start_time(self) -> float:
        """Start time of the model.

        Model times should be of type float.

        Returns
        -------
        float
            The model start time.
        """
        ...

    @abstractmethod
    def get_end_time(self) -> float:
        """End time of the model.

        Returns
        -------
        float
            The maximum model time.
        """
        ...

    @abstractmethod
    def get_time_units(self) -> str:
        """Time units of the model.

        Returns
        -------
        str
            The model time unit; e.g., `days` or `s`.

        Notes
        -----
        CSDMS uses the UDUNITS standard from Unidata.
        """
        ...

    @abstractmethod
    def get_time_step(self) -> float:
        """Return the current time step of the model.

        The model time step should be of type float.

        Returns
        -------
        float
            The time step used in model.
        """
        ...

    @abstractmethod
    def get_value(self, index: int, dest: NDArray[Any]) -> NDArray[Any]:
        """Get a copy of values of the given variable.

        This is a getter for the model, used to access the model's
        current state. It returns a *copy* of a model variable, with
        the return type, size and rank dependent on the variable.

        Parameters
        ----------
        index : int
            The index of the variable, from :func:`Bmi.get_var_index`.
        dest : ndarray
            A numpy array into which to place the values.

        Returns
        -------
        ndarray
            The same numpy array that was passed as an input buffer.
        """
        ...

    @abstractmethod
    def get_value_at_indices(
        self, index: int, dest: NDArray[Any], inds: NDArray[np.int_]
    ) -> NDArray[Any]:
        """Get values at particular indices.

        Parameters
        ----------
        index : int
            The index of the variable, from :func:`Bmi.get_var_index`.
        dest : ndarray
            A numpy array into which to place the values.
        inds : array_like
            The indices into the variable array.

        Returns
        -------
        array_like
            Value of the model variable at the given location.
        """
        ...

    @abstractmethod
    def set_value(self, index: int, src: NDArray[Any]) -> None:
        """Specify a new value for a model variable.

        This is the setter for the model, used to change the model's
        current state. It accepts, through *src*, a new value for a
        model variable, with the type, size and rank of *src*
        dependent on the variable.

        Parameters
        ----------
        index : int
            The index of the variable, from :func:`Bmi.get_var_index`.
        src : array_like
            The new value for the specified variable.
        """
        ...

    @abstractmethod
    def set_value_at_indices(
        self, index: int, inds: NDArray[np.int_], src: NDArray[Any]
    ) -> None:
        """Specify a new value for a model variable at particular indices.

        Parameters
        ----------
        index : int
            The index of the variable, from :func:`Bmi.get_var_index`.
        inds : array_like
            The indices into the variable array.
        src : array_like
            The new value for the specified variable.
        """
        ...

    # Grid information
    @abstractmethod
    def get_grid_rank(self, grid: int) -> int:
        """Get number of dimensions of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            Rank of the grid.
        """
        ...

    @abstractmethod
    def get_grid_size(self, grid: int) -> int:
        """Get the total number of elements in the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            Size of the grid.
        """
        ...

    @abstractmethod
    def get_grid_type(self, grid: int) -> str:
        """Get the grid type as a string.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        str
            Type of grid as a string.
        """
        ...

    # Uniform rectilinear
    @abstractmethod
    def get_grid_shape(self, grid: int, shape: NDArray[np.int_]) -> NDArray[np.int_]:
        """Get dimensions of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.
        shape : ndarray of int, shape *(ndim,)*
            A numpy array into which to place the shape of the grid.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the grid's shape.
        """
        ...

    @abstractmethod
    def get_grid_spacing(
        self, grid: int, spacing: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Get distance between nodes of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.
        spacing : ndarray of float, shape *(ndim,)*
            A numpy array to hold the spacing between grid rows and columns.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's spacing.
        """
        ...

    @abstractmethod
    def get_grid_origin(
        self, grid: int, origin: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Get coordinates for the lower-left corner of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.
        origin : ndarray of float, shape *(ndim,)*
            A numpy array to hold the coordinates of the lower-left corner of
            the grid.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the coordinates of the grid's
            lower-left corner.
        """
        ...

    # Non-uniform rectilinear, curvilinear
    @abstractmethod
    def get_grid_x(self, grid: int, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get coordinates of grid nodes in the x direction.

        Parameters
        ----------
        grid : int
            A grid identifier.
        x : ndarray of float, shape *(nrows,)*
            A numpy array to hold the x-coordinates of the grid node columns.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's column x-coordinates.
        """
        ...

    @abstractmethod
    def get_grid_y(self, grid: int, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get coordinates of grid nodes in the y direction.

        Parameters
        ----------
        grid : int
            A grid identifier.
        y : ndarray of float, shape *(ncols,)*
            A numpy array to hold the y-coordinates of the grid node rows.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's row y-coordinates.
        """
        ...

    @abstractmethod
    def get_grid_z(self, grid: int, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get coordinates of grid nodes in the z direction.

        Parameters
        ----------
        grid : int
            A grid identifier.
        z : ndarray of float, shape *(nlayers,)*
            A numpy array to hold the z-coordinates of the grid nodes layers.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's layer z-coordinates.
        """
        ...

    @abstractmethod
    def get_grid_node_count(self, grid: int) -> int:
        """Get the number of nodes in the grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid nodes.
        """
        ...

    @abstractmethod
    def get_grid_edge_count(self, grid: int) -> int:
        """Get the number of edges in the grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid edges.
        """
        ...

    @abstractmethod
    def get_grid_face_count(self, grid: int) -> int:
        """Get the number of faces in the grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid faces.
        """
        ...

    @abstractmethod
    def get_grid_edge_nodes(
        self, grid: int, edge_nodes: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Get the edge-node connectivity.

        Parameters
        ----------
        grid : int
            A grid identifier.
        edge_nodes : ndarray of int, shape *(2 x nnodes,)*
            A numpy array to place the edge-node connectivity. For each edge,
            connectivity is given as node at edge tail, followed by node at
            edge head.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the edge-node connectivity.
        """
        ...

    @abstractmethod
    def get_grid_face_edges(
        self, grid: int, face_edges: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Get the face-edge connectivity.

        Parameters
        ----------
        grid : int
            A grid identifier.
        face_edges : ndarray of int
            A numpy array to place the face-edge connectivity.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the face-edge connectivity.
        """
        ...

    @abstractmethod
    def get_grid_face_nodes(
        self, grid: int, face_nodes: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Get the face-node connectivity.

        Parameters
        ----------
        grid : int
            A grid identifier.
        face_nodes : ndarray of int
            A numpy array to place the face-node connectivity. For each face,
            the nodes (listed in a counter-clockwise direction) that form the
            boundary of the face.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the face-node connectivity.
        """
        ...

    @abstractmethod
    def get_grid_nodes_per_face(
        self, grid: int, nodes_per_face: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Get the number of nodes for each face.

        Parameters
        ----------
        grid : int
            A grid identifier.
        nodes_per_face : ndarray of int, shape *(nfaces,)*
            A numpy array to place the number of nodes per face.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the number of nodes per face.
        """
        ...
