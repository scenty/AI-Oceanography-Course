import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// 获取正确的图片路径（适配 GitHub Pages 子路径）
export function getImagePath(path: string): string {
  const base = import.meta.env.BASE_URL
  // 移除路径开头的斜杠，然后与 base 拼接
  const cleanPath = path.startsWith('/') ? path.slice(1) : path
  return `${base}${cleanPath}`
}
