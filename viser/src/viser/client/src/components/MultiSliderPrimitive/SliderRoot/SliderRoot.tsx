import React, { forwardRef } from "react";
import {
  Box,
  BoxProps,
  ElementProps,
  MantineColor,
  MantineRadius,
  MantineSize,
} from "@mantine/core";
import { useSliderContext } from "../Slider.context";

export interface SliderRootProps extends BoxProps, ElementProps<"div"> {
  size: MantineSize | (string & NonNullable<unknown>) | number;
  children: React.ReactNode;
  color: MantineColor | undefined;
  disabled: boolean | undefined;
  variant?: string;
  thumbSize: string | number | undefined;
  radius: MantineRadius | undefined;
}

export const SliderRoot = forwardRef<HTMLDivElement, SliderRootProps>(
  ({ size, variant, ...others }: SliderRootProps, ref) => {
    const { getStyles } = useSliderContext();

    return (
      <Box
        tabIndex={-1}
        variant={variant}
        size={size}
        ref={ref}
        {...getStyles("root")}
        {...others}
      />
    );
  },
);

SliderRoot.displayName = "@mantine/core/SliderRoot";
