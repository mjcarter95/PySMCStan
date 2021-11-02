from abc import abstractmethod, ABC

class Target_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the target distribution.

    """

    @abstractmethod
    def logpdf(self, x):
        """
        Description
        -----------
        Returns log pdf of the target distribution, evaluated at x.

        """
        pass


class Q0_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the initial proposal distribution.

    """

    @abstractmethod
    def logpdf(self, x):
        """
        Description
        -----------
        Returns log pdf of the initial proposal, evaluated at x.
        """
        pass

    @abstractmethod
    def rvs(self, size):
        """
        Description
        -----------
        Returns samples from the initial proposal.

        Parameters
        ----------
        size : size of the sample being returned
        """
        pass

class Q_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the general proposal distribution.
    """

    @abstractmethod
    def pdf(self, x, x_cond):
        """
        Description
        -----------
        Returns q(x | x_cond)
        """
        pass

    @abstractmethod
    def logpdf(self, x, x_cond):
        """
        Description
        -----------
        Returns log q(x | x_cond)
        """
        pass

    @abstractmethod
    def rvs(self, x_cond):
        """
        Description
        -----------
        Returns a single sample from the proposal, q(x | x_cond).
        """

        pass

    @abstractmethod
    def kernel_parameters(self, x, x_new):
        """
        Description
        -----------
        Set kernel parameters such that if q(x1 | x2) and L(x3 | x4), then
        x1,x2,x3,x4 are returned.
        """

        pass
