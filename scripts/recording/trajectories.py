from typing import List
import numpy as np
import transforms3d
import math
import quaternion


class EasingBase:
    limit = (0, 1)

    def __init__(self, start: float = 0, end: float = 1, duration: float = 1):
        self.start = start
        self.end = end
        self.duration = duration

    def func(self, t: float) -> float:
        raise NotImplementedError

    def ease(self, alpha: float) -> float:
        t = self.limit[0] * (1 - alpha) + self.limit[1] * alpha
        t /= self.duration
        a = self.func(t)
        return self.end * a + self.start * (1 - a)

    def __call__(self, alpha: float) -> float:
        return self.ease(alpha)


class LinearInOut(EasingBase):
    def func(self, t: float) -> float:
        return t


class QuadEaseInOut(EasingBase):
    def func(self, t: float) -> float:
        if t < 0.5:
            return 2 * t * t
        return (-2 * t * t) + (4 * t) - 1


class QuadEaseIn(EasingBase):
    def func(self, t: float) -> float:
        return t * t


class QuadEaseOut(EasingBase):
    def func(self, t: float) -> float:
        return -(t * (t - 2))


class CubicEaseIn(EasingBase):
    def func(self, t: float) -> float:
        return t * t * t


class CubicEaseOut(EasingBase):
    def func(self, t: float) -> float:
        return (t - 1) * (t - 1) * (t - 1) + 1


class CubicEaseInOut(EasingBase):
    def func(self, t: float) -> float:
        if t < 0.5:
            return 4 * t * t * t
        p = 2 * t - 2
        return 0.5 * p * p * p + 1


class QuarticEaseIn(EasingBase):
    def func(self, t: float) -> float:
        return t * t * t * t


class QuarticEaseOut(EasingBase):
    def func(self, t: float) -> float:
        return (t - 1) * (t - 1) * (t - 1) * (1 - t) + 1


class QuarticEaseInOut(EasingBase):
    def func(self, t: float) -> float:
        if t < 0.5:
            return 8 * t * t * t * t
        p = t - 1
        return -8 * p * p * p * p + 1


class QuinticEaseIn(EasingBase):
    def func(self, t: float) -> float:
        return t * t * t * t * t


class QuinticEaseOut(EasingBase):
    def func(self, t: float) -> float:
        return (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1


class QuinticEaseInOut(EasingBase):
    def func(self, t: float) -> float:
        if t < 0.5:
            return 16 * t * t * t * t * t
        p = (2 * t) - 2
        return 0.5 * p * p * p * p * p + 1


class SineEaseIn(EasingBase):
    def func(self, t: float) -> float:
        return math.sin((t - 1) * math.pi / 2) + 1


class SineEaseOut(EasingBase):
    def func(self, t: float) -> float:
        return math.sin(t * math.pi / 2)


class SineEaseInOut(EasingBase):
    def func(self, t: float) -> float:
        return 0.5 * (1 - math.cos(t * math.pi))


class CircularEaseIn(EasingBase):
    def func(self, t: float) -> float:
        return 1 - math.sqrt(1 - (t * t))


class CircularEaseOut(EasingBase):
    def func(self, t: float) -> float:
        return math.sqrt((2 - t) * t)


class CircularEaseInOut(EasingBase):
    def func(self, t: float) -> float:
        if t < 0.5:
            return 0.5 * (1 - math.sqrt(1 - 4 * (t * t)))
        return 0.5 * (math.sqrt(-((2 * t) - 3) * ((2 * t) - 1)) + 1)


class ExponentialEaseIn(EasingBase):
    def func(self, t: float) -> float:
        if t == 0:
            return 0
        return math.pow(2, 10 * (t - 1))


class ExponentialEaseOut(EasingBase):
    def func(self, t: float) -> float:
        if t == 1:
            return 1
        return 1 - math.pow(2, -10 * t)


class ExponentialEaseInOut(EasingBase):
    def func(self, t: float) -> float:
        if t == 0 or t == 1:
            return t

        if t < 0.5:
            return 0.5 * math.pow(2, (20 * t) - 10)
        return -0.5 * math.pow(2, (-20 * t) + 10) + 1


class BackEaseIn(EasingBase):
    def func(self, t: float) -> float:
        return t * t * t - t * math.sin(t * math.pi)


class BounceEaseIn(EasingBase):
    def func(self, t: float) -> float:
        return 1 - BounceEaseOut().func(1 - t)


class BounceEaseOut(EasingBase):
    def func(self, t: float) -> float:
        if t < 4 / 11:
            return 121 * t * t / 16
        elif t < 8 / 11:
            return (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0
        elif t < 9 / 10:
            return (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0
        return (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0


class BounceEaseInOut(EasingBase):
    def func(self, t: float) -> float:
        if t < 0.5:
            return 0.5 * BounceEaseIn().func(t * 2)
        return 0.5 * BounceEaseOut().func(t * 2 - 1) + 0.5


class TrajectoryInterpolator:
    DELTA_STEPS = 5000

    def __init__(
        self,
        poses: List[np.ndarray],
        easing: EasingBase = LinearInOut(),
        flyby_steps: int = 100,
        waypoint_steps: int = 100,
    ) -> None:
        self._poses = poses
        self._flyby_steps = flyby_steps
        self._waypoint_steps = waypoint_steps
        self._easing = easing

        self._trajectory = []
        for i in range(len(poses) - 1):
            T0 = poses[i]
            T1 = poses[i + 1]
            chunk = self._interpolate(
                T0,
                T1,
                easing=self._easing,
                steps=self._flyby_steps,
                endpoint=False,
            )

            chunk += [T1] * self._waypoint_steps
            self._trajectory += chunk

    def build_trajectory(self) -> List[np.ndarray]:
        return self._trajectory.copy()

    @staticmethod
    def _interpolate(
        T0: np.ndarray,
        T1: np.ndarray,
        easing: EasingBase = CubicEaseInOut(),
        steps: int = 100,
        endpoint: bool = False,
    ) -> List[np.ndarray]:
        """Interpolate between two transforms with easing function

        :param T0: Initial transform
        :type T0: np.ndarray
        :param T1: Final transform
        :type T1: np.ndarray
        :param easing: Easing function , defaults to CubicEaseInOut()
        :type easing: EasingBase, optional
        :param steps: number of interpolated steps , defaults to 100
        :type steps: int, optional
        :param endpoint: TRUE to include last transform , defaults to False
        :type endpoint: bool, optional
        :return: List of interpolated transforms
        :rtype: List[np.ndarray]
        """

        # rotations
        # x, y, z, w
        q0 = transforms3d.quaternions.mat2quat(T0[:3, :3])
        q1 = transforms3d.quaternions.mat2quat(T1[:3, :3])

        q0 = np.quaternion(q0[0], q0[1], q0[2], q0[3])
        q1 = np.quaternion(q1[0], q1[1], q1[2], q1[3])

        # positions
        p0 = T0[:3, 3]
        p1 = T1[:3, 3]

        # interpolate rotations
        qs = [
            quaternion.as_float_array(quaternion.slerp_evaluate(q0, q1, x))
            for x in np.arange(0, 1, 1.0 / TrajectoryInterpolator.DELTA_STEPS)
        ]

        # interpolate positions
        ps = np.linspace(
            p0,
            p1,
            num=TrajectoryInterpolator.DELTA_STEPS,
            endpoint=endpoint,
        )

        # generate raw poses by interpolating rotations and positions
        raw_poses = []
        for q, p in zip(qs, ps):
            T = np.eye(4)
            T[:3, :3] = transforms3d.quaternions.quat2mat(q)
            T[:3, 3] = p
            raw_poses.append(T)

        # convert raw poses to easing interpolated poses
        poses = []
        for i in range(steps):
            alpha = easing(i / steps)
            poses.append(raw_poses[int(alpha * TrajectoryInterpolator.DELTA_STEPS)])

        return poses
